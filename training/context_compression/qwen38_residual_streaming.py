"""Immutable summary shards for a growing, context-sorted training prefix.

The root descriptor never changes. Published shards form a checksum chain;
checkpoints pin that chain, so appending summaries cannot change old supervision.
GPU generation/training can alternate without resetting the epoch or optimizer.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from context_compression.qwen38_allturn_data import digest
from context_compression.qwen38_residual_data import SummaryCache, identity, summary_request
from context_compression.qwen38_residual_curriculum import load_context_order, load_ordered_response

SCHEMA = "qwen38_residual_immutable_summary_shards_v1"


def aligned_batch_end(order, start, requested_responses, accumulate_tokens):
    """End at an ordinary optimizer boundary, never flush partial accumulation."""
    if not 0 <= start < len(order) or min(requested_responses, accumulate_tokens) <= 0:
        raise ValueError("invalid summary batch boundary")
    pending = 0
    for position in range(start, len(order)):
        pending += order[position][4]
        if pending >= accumulate_tokens:
            pending = 0
            if position + 1 - start >= requested_responses:
                return position + 1
    return len(order)  # The real epoch boundary may flush a final partial batch.


def validate_boundary(order, start, end, accumulate_tokens):
    if not 0 <= start <= end <= len(order):
        raise ValueError("invalid available summary prefix")
    pending = 0
    for row in order[start:end]:
        pending += row[4]
        if pending >= accumulate_tokens:
            pending = 0
    if end < len(order) and pending:
        raise ValueError("summary boundary would change token gradient accumulation")


def create_stream_manifest(path, *, contract, order_receipt):
    path = Path(path).resolve()
    value = {"schema": SCHEMA, "contract": contract,
             "plan_sha256": order_receipt["plan_sha256"],
             "responses": order_receipt["responses"],
             "shards_dir": str(path.parent / "summary_shards"),
             "cache_sha256": None,
             "coverage": {"responses": order_receipt["responses"],
                          "trajectories": order_receipt["trajectories"], "ordinals": None}}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("stream descriptor changed")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(value["shards_dir"]).mkdir(exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
    return value


class StreamingSummaryCache:
    def __init__(self, path, *, dataset_manifest, order, order_receipt):
        self.manifest = json.loads(Path(path).read_text())
        value = self.manifest
        self.contract = value["contract"]
        if (value["schema"] != SCHEMA or value["plan_sha256"] != order_receipt["plan_sha256"]
                or value["responses"] != len(order)
                or self.contract["database_sha256"] != dataset_manifest["database_sha256"]
                or self.contract["tokenizer_files"] != dataset_manifest["tokenizer_files"]):
            raise ValueError("stream summary/data/order identity mismatch")
        self.available = 0
        self.chain = identity(value)
        self.prefixes = {0: self.chain}
        self.caches = []
        self.lookup = {}
        shards = Path(value["shards_dir"])
        try:
            for path in sorted(shards.glob("chunk-????????-????????.json")):
                receipt = json.loads(path.read_text())
                content = {k: v for k, v in receipt.items() if k != "chain_sha256"}
                start, end = receipt["start"], receipt["end"]
                stem = f"chunk-{start:08d}-{end:08d}"
                if (start != self.available or not start < end <= len(order)
                        or path.name != stem + ".json"
                        or receipt["previous_sha256"] != self.chain
                        or receipt["order_slice_sha256"] != identity(order[start:end])
                        or receipt["chain_sha256"] != identity(content)
                        or receipt["cache_sha256"] != digest(shards / (stem + ".sqlite"))):
                    raise ValueError("summary shard gap/checksum/chain mismatch")
                cache = SummaryCache(shards / (stem + ".sqlite"), self.contract, readonly=True)
                self.caches.append(cache)
                sealed = cache.db.execute("SELECT value FROM metadata WHERE key='sealed'").fetchone()
                if not sealed or json.loads(sealed[0]) != {"start": start, "end": end}:
                    raise ValueError("summary shard is not sealed")
                keys = list(cache.db.execute("SELECT key, turn FROM summaries"))
                if len(keys) != end-start or any(key in self.lookup for key in keys):
                    raise ValueError("summary shard coverage/duplicate mismatch")
                for key in keys:
                    self.lookup[key] = cache
                self.available = end
                self.chain = receipt["chain_sha256"]
                self.prefixes[end] = self.chain
        except BaseException:
            self.close()
            raise

    def get(self, event):
        cache = self.lookup.get((event.key, event.turn))
        if cache is None:
            raise ValueError(f"missing published summary: {event.key} turn {event.turn}")
        return cache.get(event)

    def receipt(self):
        return {"available_responses": self.available, "chain_sha256": self.chain}

    def verify_checkpoint_prefix(self, receipt):
        if (not receipt or self.prefixes.get(receipt["available_responses"])
                != receipt["chain_sha256"]):
            raise ValueError("checkpoint's immutable summary prefix changed or disappeared")

    def close(self):
        for cache in self.caches:
            cache.close()


def publish_shard(stream, source, database, order, end):
    start = stream.available
    if not start < end <= len(order):
        raise ValueError("invalid shard range")
    shards = Path(stream.manifest["shards_dir"])
    stem = f"chunk-{start:08d}-{end:08d}"
    destination = shards / (stem + ".sqlite")
    receipt_path = shards / (stem + ".json")
    staging = shards / (stem + ".staging.sqlite")
    if any(path.exists() for path in (destination, receipt_path, staging)):
        raise FileExistsError("refusing to overwrite a published/staged shard")
    cache = SummaryCache(staging, stream.contract)
    try:
        for row in order[start:end]:
            event, _, _ = load_ordered_response(database, row)
            cache.put(event, source.get(event)["ids"], finish_reason="stop")
        with cache.db:
            cache.db.execute("INSERT INTO metadata VALUES ('sealed', ?)",
                             (json.dumps({"start": start, "end": end}),))
        if cache.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("summary shard integrity check failed")
    finally:
        cache.close()
    os.replace(staging, destination)
    receipt = {"start": start, "end": end, "previous_sha256": stream.chain,
               "order_slice_sha256": identity(order[start:end]), "cache_sha256": digest(destination)}
    receipt["chain_sha256"] = identity(receipt)
    temporary = shards / (stem + ".staging.json")
    with temporary.open("x") as handle:
        json.dump(receipt, handle, indent=2)
    os.replace(temporary, receipt_path)  # Publication commit; readers ignore staging files.
    return receipt


def main():
    from context_compression.prepare_qwen38_residual_summaries import (
        validate_dataset, generate_one, request_json,
    )
    from context_compression.qwen38_residual_data import summary_contract
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--context-order-plan", type=Path, required=True)
    parser.add_argument("--stream-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--responses", type=int, default=2048)
    parser.add_argument("--accumulate-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    manifest, database = validate_dataset(args.manifest, args.target_model)
    order, order_receipt = load_context_order(args.context_order_plan, dataset_manifest=manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    contract = summary_contract(tokenizer, database_sha256=manifest["database_sha256"],
        tokenizer_files=manifest["tokenizer_files"], generator_model=str(args.target_model.resolve()),
        max_summary_tokens=4096)
    create_stream_manifest(args.stream_manifest, contract=contract, order_receipt=order_receipt)
    stream = StreamingSummaryCache(args.stream_manifest, dataset_manifest=manifest,
                                   order=order, order_receipt=order_receipt)
    source = SummaryCache(args.cache, contract)
    try:
        if stream.available == len(order):
            print(json.dumps({"all_summaries_complete": True}), flush=True)
            return
        end = aligned_batch_end(order, stream.available, args.responses, args.accumulate_tokens)
        info = request_json(args.base_url.rstrip("/") + "/get_model_info")
        if info.get("model_path") != str(args.target_model.resolve()):
            raise ValueError("summary server model differs from pinned frozen target")
        positions = iter(range(stream.available, end))
        pending = {}
        created = reused = 0
        exhausted = False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers:
                    position = next(positions, None)
                    if position is None:
                        exhausted = True
                        break
                    event, _, _ = load_ordered_response(database, order[position])
                    if source.get(event, missing_ok=True) is not None:
                        reused += 1
                        continue
                    future = pool.submit(generate_one, args.base_url, summary_request(event, contract),
                                         contract, context_limit=131072)
                    pending[future] = (position, event)
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    position, event = pending.pop(future)
                    try:
                        ids, reason = future.result()
                        source.put(event, ids, finish_reason=reason)
                    except Exception as error:
                        raise RuntimeError(f"summary failed at sorted position={position}: {error}") from error
                    created += 1
                    if created == 1 or created % 100 == 0:
                        print(json.dumps({"created": created, "reused": reused, "position": position,
                                          "batch_start": stream.available, "batch_end": end}), flush=True)
        receipt = publish_shard(stream, source, database, order, end)
        print(json.dumps({"published": receipt, "created": created, "reused": reused,
                          "all_summaries_complete": end == len(order)}), flush=True)
    finally:
        source.close()
        stream.close()
        database.close()


if __name__ == "__main__":
    main()
