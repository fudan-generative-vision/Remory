"""Derive length-qualified residual tasks without changing the source trajectories.

Each derived row is one independent compression event. Boundary events retain
the entire answer; short-prompt single turns contribute only intra-reasoning
continuations. Exact token slices are used throughout, without retokenization.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from context_compression.qwen38_allturn_data import SCHEMA, OPENING, digest
from context_compression.qwen38_residual_data import identity

DERIVATION_SCHEMA = "qwen38_length_filtered_history_and_reasoning_cuts_v1"


def policy(*, min_history=8192, recent_tokens=256, target_window=8192,
           min_target=256):
    if min(min_history, recent_tokens, target_window, min_target) <= 0:
        raise ValueError("all data budgets must be positive")
    if recent_tokens >= min_history or min_target > target_window:
        raise ValueError("invalid recent-tail or continuation budget")
    return {
        "schema": DERIVATION_SCHEMA, "min_history_tokens": min_history,
        "recent_tokens": recent_tokens, "target_window_tokens": target_window,
        "min_continuation_tokens": min_target,
        "history_coordinate": "compressible_tokens_excluding_system_tools_and_recent_tail",
        "boundary": "all_response_positions_with_qualified_history_full_answer",
        "intra_reasoning": "single_turn_with_short_initial_history_only",
        "cuts": "first_qualified_known_prefix_then_target_window_stride_while_in_think",
        "target_windows": "nonoverlapping_within_each_source_response_no_synthetic_eot",
        "student": "native_assistant_think_opening_plus_exact_recent_tail",
        "teacher": "unchanged_native_prefix_through_target_window",
        "summary": "compressed_history_only_with_synthetic_think_and_role_closure_at_cuts",
        "position_weighting": "original_response_absolute_token_offset",
    }


def derive_examples(example, ordinal, rules, *, native_opening, think_end_id,
                    summary_closing_ids):
    """Decide cuts from known prefix length/phase; never inspect target contents.

    The end-of-thinking marker only prevents cuts after reasoning has ended.
    Targets are raw slices. Removing/changing a future suffix cannot change any
    existing cut's history, recent tail, request key, or summary input.
    """
    minimum = rules["min_history_tokens"]
    recent = rules["recent_tokens"]
    window = rules["target_window_tokens"]
    min_target = rules["min_continuation_tokens"]
    history = []
    single = len(example["segments"]) == 1

    def make(kind, turn, context, opening, target, start, stop, **extra):
        origin = {"key": example["key"], "ordinal": ordinal, "turn": turn,
                  "kind": kind, "target_start": start, "target_stop": stop}
        return {"key": f"{example['key']}/turn{turn}/{kind}/cut{start}",
                "corpus": example["corpus"], "prefix_ids": example["prefix_ids"],
                "source": origin,
                "segments": [{"context_ids": list(context), "opening_ids": list(opening),
                              "response_ids": list(target), "target_offset": start, **extra}]}

    for turn, segment in enumerate(example["segments"]):
        history.extend(segment["context_ids"])
        opening, response = segment["opening_ids"], segment["response_ids"]
        if not history or not opening or not response:
            raise ValueError("invalid source response")
        if len(history) >= minimum:
            yield make("long_prompt" if single else "long_history", turn,
                       history, opening, response, 0, len(response))
        elif single:
            if list(opening) != list(native_opening):
                raise ValueError("intra-reasoning cut requires the pinned native think opening")
            # A normal completed response must explicitly close its thinking phase.
            # Do not silently call an unmarked long answer a reasoning trajectory.
            try:
                reasoning_end = response.index(think_end_id)
            except ValueError:
                reasoning_end = 0
            first = max(minimum, minimum - len(history) - len(opening) + recent)
            for cut in range(first, reasoning_end, window):
                stop = min(cut + window, len(response))
                if stop - cut < min_target:
                    continue
                known = history + list(opening) + response[:cut]
                source, tail = known[:-recent], known[-recent:]
                if len(source) < minimum:
                    raise AssertionError("compression source below length threshold")
                yield make("reasoning_cut", turn, source, tail, response[cut:stop], cut, stop,
                           student_opening_ids=list(opening) + tail,
                           summary_closing_ids=list(summary_closing_ids))
        history.extend(opening)
        history.extend(response)


def build(args):
    from context_compression.prepare_qwen38_residual_summaries import validate_dataset
    from transformers import AutoTokenizer

    destination = args.output.resolve()
    manifest_path = destination.with_suffix(".manifest.json")
    staging = destination.with_suffix(".building.sqlite")
    if any(p.exists() for p in (destination, manifest_path, staging)):
        raise FileExistsError("derived dataset already exists; do not overwrite")
    parent, source_db = validate_dataset(args.manifest, args.target_model)
    if "derivation" in parent:
        raise ValueError("derive from original allturn trajectories only")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    closing = encode("\n</think>\n<|im_end|>\n")
    end_ids = encode("</think>")
    if len(end_ids) != 1:
        raise ValueError("thinking end must be one native special token")
    native_opening = encode(OPENING)
    rules = policy(min_history=args.min_history_tokens, recent_tokens=args.recent_tokens,
                   target_window=args.target_window_tokens, min_target=args.min_target_tokens)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = sqlite3.connect(staging)
    output.execute("CREATE TABLE examples (ordinal INTEGER PRIMARY KEY, key TEXT UNIQUE, "
                   "corpus TEXT, responses INTEGER, tokens INTEGER, prefix_tokens INTEGER, "
                   "max_response INTEGER, payload BLOB)")
    stats, audit, selected_sources = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    n = 0
    try:
        for ordinal, payload in source_db.execute("SELECT ordinal,payload FROM examples ORDER BY ordinal"):
            example = json.loads(zlib.decompress(payload))
            corpus = example["corpus"]
            a = audit[corpus]
            a["source_trajectories"] += 1
            a["single_turn" if len(example["segments"]) == 1 else "multi_turn"] += 1
            found = False
            for derived in derive_examples(example, ordinal, rules, native_opening=native_opening,
                                            think_end_id=end_ids[0], summary_closing_ids=closing):
                segment = derived["segments"][0]
                target_len = len(segment["response_ids"])
                prefix_len = len(derived["prefix_ids"])
                history_len = len(segment["context_ids"])
                output.execute("INSERT INTO examples VALUES (?,?,?,?,?,?,?,?)", (
                    n, derived["key"], corpus, 1, target_len, prefix_len, target_len,
                    zlib.compress(json.dumps(derived, separators=(",", ":")).encode(), 1)))
                n += 1
                found = True
                selected_sources[corpus].add(ordinal)
                s = stats[corpus]
                s.update(trajectories=1, responses=1, target_tokens=target_len)
                s["max_prefix_tokens"] = max(s["max_prefix_tokens"], prefix_len)
                s["max_response_tokens"] = max(s["max_response_tokens"], target_len)
                a[derived["source"]["kind"]] += 1
                a["max_history_tokens"] = max(a["max_history_tokens"], history_len)
                a["min_history_tokens"] = min(a.get("min_history_tokens", history_len), history_len)
            if not found:
                a["excluded_trajectories"] += 1
            if (ordinal + 1) % 5000 == 0:
                output.commit()
                print(json.dumps({"source_rows": ordinal + 1, "derived_events": n}), flush=True)
        if not n:
            raise ValueError("no qualifying long-history events")
        output.commit()
        if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("derived dataset integrity failure")
    finally:
        output.close()
        source_db.close()
    os.replace(staging, destination)
    for corpus, selected in selected_sources.items():
        audit[corpus]["selected_source_trajectories"] = len(selected)
    derivation = {"rules": rules, "parent_manifest": str(args.manifest.resolve()),
                  "parent_manifest_sha256": digest(args.manifest),
                  "parent_database_sha256": parent["database_sha256"],
                  "builder_sha256": digest(Path(__file__)),
                  "native_opening_ids": native_opening, "think_end_id": end_ids[0],
                  "summary_closing_ids": closing, "audit": dict(audit)}
    manifest = {"schema": SCHEMA, "database": str(destination),
                "database_sha256": digest(destination), "stats": dict(stats),
                "tokenizer_model": parent["tokenizer_model"], "tokenizer_files": parent["tokenizer_files"],
                "data_implementation_sha256": parent["data_implementation_sha256"],
                "coverage": "derived_independent_compression_events_not_all_source_responses",
                "mix": "one_pass_each_derived_event_no_oversampling",
                "derivation": derivation, "derivation_sha256": identity(derivation)}
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"complete": True, "manifest": str(manifest_path),
                      "stats": dict(stats), "audit": dict(audit)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-history-tokens", type=int, default=8192)
    parser.add_argument("--recent-tokens", type=int, default=256)
    parser.add_argument("--target-window-tokens", type=int, default=8192)
    parser.add_argument("--min-target-tokens", type=int, default=256)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
