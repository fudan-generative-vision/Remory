"""Append-safe, exhaustive xhigh/coding assistant-continuation data.

No WM round eligibility, no teacher-context ceiling, no sampled target window.
Tokenization follows native generation boundaries: prompt and generated suffix
are encoded separately, never decode/re-encoded. All generated turns survive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from context_compression.qwen38_coding_trajectory_data import _normalize_message

SCHEMA = "qwen38_xhigh_coding_allturn_exact_tokens_v1"
OPENING = "<|im_start|>assistant\n<think>\n"
_TOKENIZER: Any = None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_row(row: dict, corpus: str, tokenizer: Any) -> dict:
    if corpus == "xhigh":
        if row.get("status") != "success":
            raise ValueError("xhigh row is not successful")
        raw = row["conversations"]
        indices = [i for i, m in enumerate(raw) if m.get("role") == "assistant"]
    elif corpus == "coding":
        if row.get("validation", {}).get("valid") is not True:
            raise ValueError("coding row failed validation")
        if row.get("truncated", False):
            raise ValueError("truncated coding row in accepted shard")
        raw = row["messages"]
        start = row.get("generation_start_message_index", 2)
        indices = row.get("generated_assistant_message_indices")
        expected = [
            i for i, m in enumerate(raw) if i >= start and m.get("role") == "assistant"
        ]
        if indices is None:
            indices = expected
        if indices != expected:
            raise ValueError(
                "generated target indices omit or misidentify assistant turns"
            )
    else:
        raise ValueError("only xhigh and coding are allowed")
    if not indices:
        raise ValueError("no assistant targets")
    messages = [_normalize_message(m, message_index=i) for i, m in enumerate(raw)]
    tools = row.get("tools") or []

    def template(ms: list, generation: bool = False) -> str:
        return tokenizer.apply_chat_template(
            ms,
            tools=tools,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=True,
            preserve_thinking=True,
            reasoning_effort="xhigh",
        )

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # Render system/tool definitions separately without searching inside user text.
    system = messages[:1] if messages[0]["role"] == "system" else []
    sentinel = "ALLTURN_PREFIX_SENTINEL"
    dummy = template(system + [{"role": "user", "content": sentinel}])
    suffix = "<|im_start|>user\n" + sentinel + "<|im_end|>\n"
    if not dummy.endswith(suffix):
        raise ValueError("native system-prefix template changed")
    prefix = dummy[: -len(suffix)]
    previous = prefix
    segments = []
    for index in indices:
        m = messages[index]
        if not (
            m["content"].strip()
            or str(m.get("reasoning_content") or "").strip()
            or m.get("tool_calls")
        ):
            raise ValueError("empty assistant target")
        prompt = template(messages[:index], True)
        complete = template(messages[: index + 1])
        if (
            not prompt.startswith(previous)
            or not prompt.endswith(OPENING)
            or not complete.startswith(prompt)
        ):
            raise ValueError("native incremental chat template is not prefix stable")
        if not complete.endswith("<|im_end|>\n"):
            raise ValueError("assistant continuation has no end-of-message token")
        response = complete[len(prompt) : -1]  # score im_end; newline is later context
        context = prompt[len(previous) : -len(OPENING)]
        segments.append(
            {
                "assistant_index": index,
                "context_ids": encode(context),
                "opening_ids": encode(OPENING),
                "response_ids": encode(response),
            }
        )
        previous = complete[:-1]
    return {
        "key": f"{corpus}:{row['id']}",
        "corpus": corpus,
        "prefix_ids": encode(prefix),
        "segments": segments,
    }


def event_plan(
    example: dict, *, block: int = 1024, window: int = 256
) -> Iterator[dict]:
    """C(M + incremental history) before each response, then causal streaming.

    Short boundary folds reserve one 1024-token block, without inserting pad
    token IDs into the reader. Previous soft rows count as physical source rows.
    Within long responses, consumed tokens are folded in chronological blocks.
    Every response token is scored exactly once, with response-global position.
    """
    if block <= 0 or window <= 0 or block % window:
        raise ValueError("block must be a positive multiple of window")
    pending: list[int] = []
    for turn, segment in enumerate(example["segments"]):
        pending += segment["context_ids"]
        while pending:
            chunk, pending = pending[:block], pending[block:]
            yield {"kind": "fold", "ids": chunk, "padded_new_tokens": block}
        pending = list(segment["opening_ids"])
        response = segment["response_ids"]
        for start in range(0, len(response), window):
            target = response[start : start + window]
            yield {
                "kind": "score",
                "prefix_ids": list(pending),
                "target_ids": target,
                "position": start,
                "turn": turn,
            }
            pending += target
            # Do not build an unused memory after the last response.
            if start + window < len(response):
                while len(pending) >= block:
                    chunk, pending = pending[:block], pending[block:]
                    yield {"kind": "fold", "ids": chunk, "padded_new_tokens": block}


def _init_worker(model: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)


def _prepare_one(item: tuple[str, bytes]) -> tuple:
    corpus, line = item
    row = json.loads(line)
    example = render_row(row, corpus, _TOKENIZER)
    responses = [len(s["response_ids"]) for s in example["segments"]]
    return (
        example["key"],
        corpus,
        len(responses),
        sum(responses),
        len(example["prefix_ids"]),
        max(responses),
        zlib.compress(json.dumps(example, separators=(",", ":")).encode(), 1),
    )


def prepare(args: argparse.Namespace) -> None:
    destination = args.output.resolve()
    if destination.exists() or destination.with_suffix(".manifest.json").exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".building.sqlite")
    if staging.exists():
        raise FileExistsError(staging)
    sources = [("xhigh", args.xhigh.resolve())] + [
        ("coding", p.resolve())
        for p in sorted(args.coding_dir.glob("qwen38_27b_trajectories.part-*.jsonl"))
    ]
    if len(sources) < 2:
        raise ValueError("coding accepted shards not found")
    # Freeze each observed byte size before starting any tokenization.
    frozen = [(c, p, p.stat().st_size) for c, p in sources]
    receipts = []

    def rows() -> Iterator[tuple[str, bytes]]:
        seen: dict[str, str] = {}
        for corpus, path, size in frozen:
            h = hashlib.sha256()
            count = complete_bytes = duplicate_rows = 0
            with path.open("rb") as handle:
                while handle.tell() < size:
                    line = handle.readline(size - handle.tell())
                    if not line.endswith(b"\n"):
                        break
                    h.update(line)
                    complete_bytes += len(line)
                    count += 1
                    row = json.loads(line)
                    key = f"{corpus}:{row['id']}"
                    row_hash = hashlib.sha256(line).hexdigest()
                    if key in seen:
                        if seen[key] != row_hash:
                            raise ValueError(f"conflicting duplicate {key}")
                        duplicate_rows += 1
                        continue
                    seen[key] = row_hash
                    yield corpus, line
            receipts.append(
                {
                    "corpus": corpus,
                    "path": str(path),
                    "observed_bytes": size,
                    "complete_bytes": complete_bytes,
                    "partial_bytes": size - complete_bytes,
                    "sha256": h.hexdigest(),
                    "rows": count,
                    "duplicate_rows": duplicate_rows,
                }
            )

    database = sqlite3.connect(staging)
    database.execute(
        "CREATE TABLE examples (ordinal INTEGER PRIMARY KEY, key TEXT UNIQUE, corpus TEXT, responses INTEGER, tokens INTEGER, prefix_tokens INTEGER, max_response INTEGER, payload BLOB)"
    )
    stats: dict[str, Counter] = {"xhigh": Counter(), "coding": Counter()}
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_init_worker, initargs=(str(args.target_model),)
    ) as pool:
        for n, record in enumerate(pool.imap(_prepare_one, rows(), chunksize=16), 1):
            database.execute(
                "INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (n - 1, *record)
            )
            _, corpus, responses, tokens, prefix, max_response, _ = record
            stat = stats[corpus]
            stat.update(
                {"trajectories": 1, "responses": responses, "target_tokens": tokens}
            )
            stat["max_prefix_tokens"] = max(stat["max_prefix_tokens"], prefix)
            stat["max_response_tokens"] = max(stat["max_response_tokens"], max_response)
            if n % 1000 == 0:
                database.commit()
                print(json.dumps({"prepared": n, "stats": stats}), flush=True)
    database.commit()
    check = database.execute("PRAGMA integrity_check").fetchone()[0]
    if check != "ok":
        raise RuntimeError(check)
    database.close()
    os.replace(staging, destination)
    tokenizer_files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    ):
        path = args.target_model / name
        if path.exists():
            tokenizer_files[name] = digest(path)
    manifest = {
        "schema": SCHEMA,
        "database": str(destination),
        "database_sha256": digest(destination),
        "sources": receipts,
        "stats": stats,
        "tokenizer_model": str(args.target_model.resolve()),
        "tokenizer_files": tokenizer_files,
        "coverage": "all_valid_generated_assistant_turns_all_tokens",
        "mix": "one_pass_each_corpus_no_oversampling",
        "block_tokens": 1024,
        "window_tokens": 256,
        "minimum_tool_rounds": 0,
        "data_implementation_sha256": digest(Path(__file__)),
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "complete": True,
                "manifest": str(destination.with_suffix(".manifest.json")),
                "stats": stats,
            }
        ),
        flush=True,
    )


def open_database(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def load_example(database: sqlite3.Connection, ordinal: int) -> dict:
    row = database.execute(
        "SELECT payload FROM examples WHERE ordinal=?", (int(ordinal),)
    ).fetchone()
    if row is None:
        raise IndexError(ordinal)
    return json.loads(zlib.decompress(row[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xhigh", type=Path, required=True)
    parser.add_argument("--coding-dir", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
