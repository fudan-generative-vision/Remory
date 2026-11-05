"""Explicit, resumable offline summary generation via a user-selected SGLang server.

Does not launch a server, claim GPUs, regenerate answers, truncate histories,
retry indefinitely, or start training. --check-only makes NO generation requests.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from context_compression.qwen38_allturn_data import SCHEMA as DATA_SCHEMA, digest, load_example, open_database
from context_compression.qwen38_residual_data import (
    SummaryCache, residual_events, summary_contract, summary_request,
)


def validate_dataset(manifest_path, model_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["schema"] != DATA_SCHEMA:
        raise ValueError("wrong allturn data schema")
    if digest(Path(manifest["database"])) != manifest["database_sha256"]:
        raise ValueError("allturn database checksum mismatch")
    for name, checksum in manifest["tokenizer_files"].items():
        if digest(Path(model_path) / name) != checksum:
            raise ValueError(f"tokenizer identity changed: {name}")
    if manifest["data_implementation_sha256"] != digest(Path(__file__).with_name("qwen38_allturn_data.py")):
        raise ValueError("data implementation identity changed")
    if "derivation" in manifest:
        from context_compression.qwen38_long_history_data import DERIVATION_SCHEMA
        from context_compression.qwen38_residual_data import identity
        derived = manifest["derivation"]
        if (derived["rules"]["schema"] != DERIVATION_SCHEMA
                or manifest.get("derivation_sha256") != identity(derived)
                or derived["builder_sha256"] != digest(Path(__file__).with_name("qwen38_long_history_data.py"))):
            raise ValueError("long-history derivation identity changed")
    db = open_database(Path(manifest["database"]))
    count, responses, tokens = db.execute("SELECT COUNT(*), SUM(responses), SUM(tokens) FROM examples").fetchone()
    if (count, responses, tokens) != tuple(sum(s[key] for s in manifest["stats"].values())
                                          for key in ("trajectories", "responses", "target_tokens")):
        raise ValueError("allturn manifest coverage mismatch")
    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise ValueError("allturn database integrity check failed")
    return manifest, db


def request_json(url, payload=None, *, timeout=3600):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def generate_one(base_url, request, contract, *, context_limit):
    if len(request["input_ids"]) + contract["max_summary_tokens"] + 1 > context_limit:
        raise ValueError("summary request exceeds server context; no history truncation")
    result = request_json(base_url.rstrip("/") + "/generate", {
        "input_ids": request["input_ids"], "stream": False,
        "sampling_params": {"temperature": 0.0,
            "max_new_tokens": contract["max_summary_tokens"] + 1,
            "stop_token_ids": [contract["end_token_id"]],
            "skip_special_tokens": False, "no_stop_trim": True},
    })
    if not isinstance(result, dict) or not isinstance(result.get("output_ids"), list):
        raise ValueError("SGLang must return exact output_ids; text retokenization is not accepted")
    finish = result.get("meta_info", {}).get("finish_reason")
    kind = finish.get("type") if isinstance(finish, dict) else finish
    if kind != "stop":
        raise ValueError(f"summary incomplete: finish_reason={finish}")
    return result["output_ids"], kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--base-url", help="Explicit SGLang server URL; no server is started")
    parser.add_argument("--max-summary-tokens", type=int, default=512)
    parser.add_argument("--context-limit", type=int, default=262144)
    parser.add_argument("--ordinals", default="", help="Canary-only subset; cannot be used for full training")
    parser.add_argument("--max-new-summaries", type=int, default=0, help="Leave an unsealed resumable cache after this many requests")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Bounded concurrent local SGLang requests")
    args = parser.parse_args()
    if args.max_new_summaries < 0 or args.context_limit <= 0 or args.workers <= 0:
        parser.error("invalid generation budget")
    manifest, db = validate_dataset(args.manifest, args.target_model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    contract = summary_contract(tokenizer, database_sha256=manifest["database_sha256"],
        tokenizer_files=manifest["tokenizer_files"], generator_model=str(args.target_model.resolve()),
        max_summary_tokens=args.max_summary_tokens)
    count = db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
    selected = [int(x) for x in args.ordinals.split(",") if x] or None
    if selected is not None and (len(set(selected)) != len(selected) or min(selected) < 0 or max(selected) >= count):
        raise ValueError("invalid canary ordinals")
    if args.check_only:
        print(json.dumps({"validated": True, "trajectories": count, "stats": manifest["stats"],
                          "selected_ordinals": selected, "summary_contract": contract}))
        db.close()
        return
    if args.output_manifest.exists():
        raise FileExistsError("summary manifest already exists; do not overwrite a sealed dataset")
    if not args.base_url:
        parser.error("--base-url is required to generate summaries")
    info = request_json(args.base_url.rstrip("/") + "/get_model_info")
    if info.get("model_path") != str(args.target_model.resolve()):
        raise ValueError("summary server model_path differs from the pinned local target")
    cache = SummaryCache(args.cache, contract)
    created = skipped = 0
    try:
        events = ((ordinal, event) for ordinal in (selected if selected is not None else range(count))
                  for event in residual_events(load_example(db, ordinal)))
        pending = {}
        exhausted = limited = False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers:
                    if args.max_new_summaries and created + len(pending) >= args.max_new_summaries:
                        exhausted = limited = True
                        break
                    item = next(events, None)
                    if item is None:
                        exhausted = True
                        break
                    ordinal, event = item
                    if cache.get(event, missing_ok=True) is not None:
                        skipped += 1
                        continue
                    future = pool.submit(generate_one, args.base_url, summary_request(event, contract),
                                         contract, context_limit=args.context_limit)
                    pending[future] = (ordinal, event)
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)  # Event-driven; no polling.
                for future in done:
                    ordinal, event = pending.pop(future)
                    try:
                        ids, reason = future.result()
                        cache.put(event, ids, finish_reason=reason)
                    except Exception as error:
                        raise RuntimeError(f"summary failed at ordinal={ordinal} turn={event.turn}: {error}") from error
                    created += 1
                    if created == 1 or created % 100 == 0:
                        print(json.dumps({"created": created, "reused": skipped, "ordinal": ordinal,
                                          "turn": event.turn}), flush=True)
        if limited:
            print(json.dumps({"sealed": False, "created": created, "reused": skipped,
                              "reason": "explicit request budget; resume with same cache"}), flush=True)
            return
        receipt = cache.seal(db, ordinals=selected)
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement of a training snapshot.
        with args.output_manifest.open("x") as handle:
            json.dump(receipt, handle, indent=2)
        print(json.dumps({"sealed": True, "coverage": receipt["coverage"],
                          "manifest": str(args.output_manifest)}), flush=True)
    finally:
        cache.close()
        db.close()


if __name__ == "__main__":
    main()
