"""Prepare causal Stage-2 summaries alongside Stage 1 using frozen Qwen servers.

Uses the pinned Codex CLI compact prompt verbatim. Multiple --base-url values
distribute requests across independently reserved servers, with one cache writer.
Never starts a server or a trainer. --check-only sends no generation requests.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from datetime import datetime, timezone
from functools import partial
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from context_compression.codex_compact_v0149 import (
    HANDOFF_FIELDS, summary_contract, structured_summary_contract, verify_binary,
)
from context_compression.prepare_qwen38_residual_summaries import (
    generate_one, request_json, validate_dataset,
)
from context_compression.qwen38_allturn_data import load_example
from context_compression.qwen38_residual_data import (
    SummaryCache, residual_events, summary_request,
)


def generate_handoff(base_url, request, contract, *, context_limit, tokenizer):
    """Constrained handoff JSON; never accept truncation, invalid JSON or no content."""
    if "output_schema" not in contract:
        return generate_one(base_url, request, contract, context_limit=context_limit)
    if len(request["input_ids"]) + contract["max_summary_tokens"] + 1 > context_limit:
        raise ValueError("summary request exceeds server context; no history truncation")
    result = request_json(base_url.rstrip("/") + "/generate", {
        "input_ids": request["input_ids"], "stream": False,
        "sampling_params": {"temperature": 0.0,
            "max_new_tokens": contract["max_summary_tokens"] + 1,
            "stop_token_ids": [contract["end_token_id"]],
            "skip_special_tokens": False, "no_stop_trim": True,
            "json_schema": json.dumps(contract["output_schema"])}
    })
    if not isinstance(result, dict) or not isinstance(result.get("output_ids"), list):
        raise ValueError("SGLang must return exact output_ids")
    finish = result.get("meta_info", {}).get("finish_reason")
    kind = finish.get("type") if isinstance(finish, dict) else finish
    if kind != "stop":
        raise ValueError(f"summary incomplete: finish_reason={finish}")
    ids = list(result["output_ids"])
    if ids and ids[-1] == contract["end_token_id"]:
        ids.pop()
    if not ids or len(ids) > contract["max_summary_tokens"]:
        raise ValueError("invalid summary length")
    item = json.loads(tokenizer.decode(ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False))
    if not isinstance(item, dict) or set(item) != set(HANDOFF_FIELDS):
        raise ValueError("summary lacks required handoff fields")
    if any(not isinstance(values, list) or len(values) > 12
           or any(not isinstance(v, str) or not v.strip() for v in values)
           for values in item.values()):
        raise ValueError("invalid handoff field values")
    if any(len(item[name]) < spec.get("minItems", 0)
           for name, spec in contract["output_schema"]["properties"].items()):
        raise ValueError("empty required handoff field")
    if not any(item.values()):
        raise ValueError("empty handoff")
    return ids, kind


def generate_validated(base_url, request, contract, *, context_limit, generate, attempts):
    """Bounded retries, with identical prompt/decoding contract on every attempt."""
    for attempt in range(1, attempts + 1):
        try:
            ids, reason = generate(base_url, request, contract, context_limit=context_limit)
            if reason != "stop":
                raise ValueError("summary was not completed naturally")
            if isinstance(ids, list) and ids and ids[-1] == contract["end_token_id"]:
                ids = ids[:-1]
            if (not isinstance(ids, list) or not 0 < len(ids) <= contract["max_summary_tokens"]
                    or any(type(x) is not int or x < 0 for x in ids)
                    or contract["end_token_id"] in ids):
                raise ValueError("invalid/empty/oversized summary token IDs")
            return ids, reason
        except Exception as error:
            if attempt == attempts:
                raise RuntimeError(f"generation failed after {attempts} attempt(s): {error}") from error


def fill_cache(events, cache, contract, endpoints, *, workers=4, limit=0,
               generate=generate_one, report=None, keep_going=False, max_attempts=1,
               on_failure=None, max_consecutive_failures=16):
    """Main thread owns SQLite; every endpoint has a bounded number of slots.

    By default, stop submitting on a failure and drain successful in-flight work.
    keep_going records failed requests separately and continues other samples.
    A run with holes MUST NOT seal; rerunning retries missing rows. Cache/storage
    failures remain fatal. Repeated failures trip a bounded circuit breaker.
    """
    if workers <= 0 or limit < 0 or not endpoints:
        raise ValueError("invalid summary worker/budget configuration")
    if len({url for url, _ in endpoints}) != len(endpoints):
        raise ValueError("duplicate summary endpoint")
    if min(max_attempts, max_consecutive_failures) <= 0 or (keep_going and not callable(on_failure)):
        raise ValueError("keep-going requires a failure recorder and positive retry/failure budgets")
    available = deque(endpoint for _ in range(workers) for endpoint in endpoints)
    events = iter(events)
    pending = {}
    created = reused = 0
    failed = consecutive_failures = 0
    exhausted = False
    failure = None
    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        while pending or not exhausted:
            while available and not exhausted and failure is None:
                if limit and created + len(pending) >= limit:
                    exhausted = True
                    break
                event = next(events, None)
                if event is None:
                    exhausted = True
                    break
                if cache.get(event, missing_ok=True) is not None:
                    reused += 1
                    continue
                endpoint = available.popleft()
                url, context_limit = endpoint
                future = pool.submit(generate_validated, url, summary_request(event, contract),
                                     contract, context_limit=context_limit,
                                     generate=generate, attempts=max_attempts)
                pending[future] = (event, endpoint)
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                event, endpoint = pending.pop(future)
                available.append(endpoint)
                try:
                    ids, reason = future.result()
                except Exception as error:
                    failed += 1
                    consecutive_failures += 1
                    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                              "key": event.key, "turn": event.turn,
                              "request_sha256": summary_request(event, contract)["request_sha256"],
                              "endpoint": endpoint[0], "attempts": max_attempts, "error": str(error)}
                    if on_failure:
                        on_failure(record)
                    if report:
                        report({"phase": "summary_failed", "failed": failed, **record})
                    if (not keep_going or consecutive_failures >= max_consecutive_failures) and failure is None:
                        failure = RuntimeError(
                            f"summary failed at {event.key} turn={event.turn}: {error}")
                        failure.__cause__ = error
                        exhausted = True
                    continue
                # Fail closed for cache corruption, disk errors or writer bugs;
                # these are never treated as isolated bad generation samples.
                try:
                    cache.put(event, ids, finish_reason=reason)
                except Exception as error:
                    if failure is None:
                        failure = RuntimeError(f"summary cache write failed: {error}")
                        failure.__cause__ = error
                    exhausted = True
                    continue
                created += 1
                consecutive_failures = 0
                if report and (created == 1 or created % 100 == 0):
                    report({"created": created, "reused": reused,
                            "key": event.key, "turn": event.turn,
                            **({"failed": failed} if keep_going else {})})
    if failure is not None:
        raise failure
    return {"created": created, "reused": reused,
            **({"failed": failed} if keep_going else {}),
            # Hitting the explicit budget leaves an unsealed cache, even if it
            # happens to coincide with the last event. Resume to verify coverage.
            "limited": bool(limit and created >= limit)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--base-url", action="append", default=[],
                        help="Repeat for independent SGLang replicas")
    parser.add_argument("--workers", type=int, default=4, help="Requests per server")
    parser.add_argument("--max-summary-tokens", type=int, default=4096)
    parser.add_argument("--summary-format", choices=("json", "text"), default="json",
                        help="JSON handoff grammar by default; prompt bytes are identical")
    parser.add_argument("--context-limit", type=int, default=262144)
    parser.add_argument("--max-new-summaries", type=int, default=0)
    parser.add_argument("--ordinals", default="", help="Canary-only trajectory subset")
    parser.add_argument("--codex-binary", type=Path,
                        help="Optionally verify the local reproduction binary")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true",
                        help="Record failed generations and continue; never seal incomplete coverage")
    parser.add_argument("--max-attempts", type=int, default=1,
                        help="Bounded attempts per request with identical generation parameters")
    parser.add_argument("--max-consecutive-failures", type=int, default=16,
                        help="Stop after repeated failures even with --keep-going")
    parser.add_argument("--failure-log", type=Path,
                        help="Append-only JSONL; defaults to CACHE.failures.jsonl")
    parser.add_argument("--allow-unfiltered-data", action="store_true",
                        help="Legacy experiments only: permit allturn data without length/cut derivation")
    args = parser.parse_args(argv)
    if min(args.workers, args.max_summary_tokens, args.context_limit) <= 0 or args.max_new_summaries < 0:
        parser.error("invalid generation budget")
    if min(args.max_attempts, args.max_consecutive_failures) <= 0:
        parser.error("retry/failure budgets must be positive")
    if args.codex_binary:
        verify_binary(args.codex_binary)
    manifest, db = validate_dataset(args.manifest, args.target_model)
    try:
        if "derivation" not in manifest and not args.allow_unfiltered_data:
            raise ValueError("prepare length-filtered data with residual-data first; "
                             "unfiltered all-response summaries require --allow-unfiltered-data")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
        make_contract = structured_summary_contract if args.summary_format == "json" else summary_contract
        contract = make_contract(tokenizer, database_sha256=manifest["database_sha256"],
                                    tokenizer_files=manifest["tokenizer_files"],
                                    generator_model=str(args.target_model.resolve()),
                                    max_summary_tokens=args.max_summary_tokens)
        if "derivation" in manifest:
            contract["data_derivation_sha256"] = manifest["derivation_sha256"]
            contract["compression_event_policy"] = manifest["derivation"]["rules"]
        count = db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
        selected = [int(x) for x in args.ordinals.split(",") if x] or None
        if selected is not None and (len(set(selected)) != len(selected) or min(selected) < 0 or max(selected) >= count):
            raise ValueError("invalid canary ordinals")
        if args.check_only:
            print(json.dumps({"validated": True, "trajectories": count, "stats": manifest["stats"],
                              "selected_ordinals": selected, "summary_contract": contract}))
            return
        if args.output_manifest.exists():
            raise FileExistsError("summary manifest already exists; do not overwrite sealed data")
        urls = [url.rstrip("/") for url in args.base_url]
        if not urls or len(set(urls)) != len(urls):
            parser.error("supply one or more distinct --base-url values")
        endpoints = []
        for url in urls:
            info = request_json(url + "/get_model_info")
            if info.get("model_path") != str(args.target_model.resolve()):
                raise ValueError("summary server model differs from pinned target")
            context_limit = info.get("context_length", args.context_limit)
            if type(context_limit) is not int or context_limit <= 0:
                raise ValueError("summary server returned an invalid context length")
            endpoints.append((url, min(args.context_limit, context_limit)))
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        # One process writes this resumable cache, regardless of endpoint count.
        with args.cache.with_suffix(args.cache.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.output_manifest.exists():
                raise FileExistsError("summary manifest was sealed by another process")
            cache = SummaryCache(args.cache, contract)
            try:
                if cache.db.execute("SELECT 1 FROM metadata WHERE key='sealed'").fetchone():
                    raise ValueError("cache is already sealed; preserve its original manifest")
                events = (event for ordinal in (selected if selected is not None else range(count))
                          for event in residual_events(load_example(db, ordinal)))
                failure_path = args.failure_log or args.cache.with_suffix(".failures.jsonl")
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                with failure_path.open("a") as failure_log:
                    def record_failure(row):
                        failure_log.write(json.dumps(row) + "\n")
                        failure_log.flush()
                    result = fill_cache(events, cache, contract, endpoints, workers=args.workers,
                                        limit=args.max_new_summaries,
                                        generate=partial(generate_handoff, tokenizer=tokenizer),
                                        report=lambda row: print(json.dumps(row), flush=True),
                                        keep_going=args.keep_going, max_attempts=args.max_attempts,
                                        max_consecutive_failures=args.max_consecutive_failures,
                                        on_failure=record_failure)
                if result.get("failed"):
                    print(json.dumps({"sealed": False, **result,
                                      "reason": "failed requests remain missing; retry before sealing",
                                      "failure_log": str(failure_path)}), flush=True)
                    raise SystemExit(2)
                if result["limited"]:
                    print(json.dumps({"sealed": False, **result}), flush=True)
                    return
                receipt = cache.seal(db, ordinals=selected)
                args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
                with args.output_manifest.open("x") as handle:
                    json.dump(receipt, handle, indent=2)
                print(json.dumps({"sealed": True, **result, "coverage": receipt["coverage"]}), flush=True)
            finally:
                cache.close()
    finally:
        db.close()


if __name__ == "__main__":
    main()
