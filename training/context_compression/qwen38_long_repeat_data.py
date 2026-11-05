"""Auditable long-document repeat data, tokenized once with the frozen reader.

One contiguous 8K--64K window per unique document; no cross-document packing.
Document hash determines the held-out split before window selection.
"""
from __future__ import annotations
import argparse
import array
import collections
import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

SCHEMA = "qwen38_long_document_repeat_v1"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "chat_template.jinja")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()

def tokenizer_receipt(path):
    return {name: sha256(Path(path) / name) for name in TOKENIZER_FILES
            if (Path(path) / name).is_file()}

def encode_ids(ids):
    a = array.array("I", ids)
    if a.itemsize != 4:
        raise RuntimeError("uint32 is not 4 bytes")
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()

def decode_ids(blob):
    a = array.array("I")
    a.frombytes(blob)
    if sys.byteorder != "little":
        a.byteswap()
    return tuple(a)

def split_for(text_sha):
    return "validation" if int(text_sha[:16], 16) % 100 == 0 else "train"

def bucket(width):
    return 65536 if width >= 65536 else 32768 if width >= 32768 else 16384 if width >= 16384 else 8192

def training_order(db, seed, epoch):
    # Mix all observed lengths from the first epoch; every example appears once.
    rows = db.execute("SELECT id, token_count FROM examples WHERE split='train' ORDER BY id").fetchall()
    groups = collections.defaultdict(list)
    for key, width in rows:
        groups[bucket(width)].append(key)
    rng = random.Random(seed + epoch)
    for values in groups.values():
        rng.shuffle(values)
    # Place each bucket evenly across the epoch. Exhausting the rare longest
    # bucket at the start would leave most of the run with only short examples.
    schedule = [(index / len(values), length, key)
                for length, values in groups.items()
                for index, key in enumerate(values)]
    order = [key for _, _, key in sorted(schedule)]
    if len(order) != len(set(order)) or len(order) != len(rows):
        raise RuntimeError("invalid epoch coverage")
    return order

def load_example(db, key):
    row = db.execute("SELECT document_sha256, token_start, original_tokens, token_count, tokens FROM examples WHERE id=?", (key,)).fetchone()
    if row is None:
        raise KeyError(key)
    digest, start, original, count, blob = row
    ids = decode_ids(blob)
    if len(ids) != count or start < 0 or start + count > original:
        raise ValueError("invalid token window")
    return dict(id=key, document_sha256=digest, token_start=start,
                original_tokens=original, token_ids=ids)

def load_dataset(manifest_path, target_model):
    manifest_path = Path(manifest_path).resolve()
    m = json.loads(manifest_path.read_text())
    digest = m.pop("manifest_sha256")
    if canonical(m) != digest or m["schema"] != SCHEMA:
        raise ValueError("long-repeat manifest identity mismatch")
    m["manifest_sha256"] = digest
    db_path = manifest_path.parent / m["database"]
    if sha256(db_path) != m["database_sha256"] or tokenizer_receipt(target_model) != m["tokenizer"]:
        raise ValueError("long-repeat data/tokenizer changed")
    db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    counts = dict(db.execute("SELECT split, COUNT(*) FROM examples GROUP BY split"))
    if counts != m["counts"]:
        raise ValueError("long-repeat coverage mismatch")
    return m, db

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-receipt", type=Path, required=True)
    p.add_argument("--target-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--minimum-tokens", type=int, default=8192)
    p.add_argument("--maximum-tokens", type=int, default=65536)
    p.add_argument("--minimum-edu-score", type=float, default=2.0)
    args = p.parse_args()
    if not 1 <= args.minimum_tokens <= args.maximum_tokens <= 65536:
        p.error("invalid source length")
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['RAYON_NUM_THREADS'] = '3'
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source = json.loads(args.source_receipt.read_text())
    if sha256(args.source) != source["sha256"]:
        raise ValueError("raw source hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "repeat.sqlite"
    if path.exists():
        raise FileExistsError(path)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE examples(id INTEGER PRIMARY KEY, document_id TEXT, document_sha256 TEXT UNIQUE, split TEXT, token_start INT, original_tokens INT, token_count INT, tokens BLOB, url TEXT, edu_score REAL)")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    stats = collections.Counter()
    seen = set()
    cols = ["text", "id", "url", "full_doc_lid", "full_doc_lid_score", "is_truncated", "token_count", "fw_edu_scores"]
    for batch in pq.ParquetFile(args.source).iter_batches(batch_size=512, columns=cols):
        eligible = []
        for r in batch.to_pylist():
            stats["scanned_documents"] += 1
            scores = r["fw_edu_scores"] or []
            if (r["full_doc_lid"] != "eng_Latn" or r["full_doc_lid_score"] < .95
                or r["is_truncated"] or not scores
                or sum(scores) / len(scores) < args.minimum_edu_score
                or r["token_count"] < args.minimum_tokens // 2):
                continue
            stats["tokenized_documents"] += 1
            digest = hashlib.sha256(r["text"].encode()).hexdigest()
            if digest in seen:
                stats["duplicate_texts"] += 1
                continue
            seen.add(digest)
            eligible.append((r, digest))
        tokenized = tokenizer([r["text"] for r, digest in eligible], add_special_tokens=False,
                              return_attention_mask=False)["input_ids"] if eligible else []
        for (r, digest), ids in zip(eligible, tokenized):
            if len(ids) < args.minimum_tokens:
                continue
            original = len(ids)
            width = min(original, args.maximum_tokens)
            start = int(digest[16:32], 16) % (original - width + 1)
            split = split_for(digest)
            db.execute("INSERT INTO examples(document_id,document_sha256,split,token_start,original_tokens,token_count,tokens,url,edu_score) VALUES(?,?,?,?,?,?,?,?,?)", (r["id"], digest, split, start, original, width, encode_ids(ids[start:start+width]), r["url"], sum(scores)/len(scores)))
            stats[split + "_tokens"] += width
            stats[split + "_documents"] += 1
            if stats["train_documents"] % 100 == 0:
                db.commit()
                print(json.dumps(dict(stats)), flush=True)
    db.commit()
    counts = dict(db.execute("SELECT split, COUNT(*) FROM examples GROUP BY split"))
    hist = collections.Counter((s, bucket(n)) for s, n in db.execute("SELECT split,token_count FROM examples"))
    if counts.get("train", 0) < 100 or counts.get("validation", 0) < 1 or not any(n >= 32768 and s == "train" for s, n in hist):
        raise ValueError("insufficient real long-document coverage")
    db.execute("VACUUM")
    db.close()
    m = dict(schema=SCHEMA, source=source, database=path.name, database_sha256=sha256(path),
             tokenizer=tokenizer_receipt(args.target_model), counts=counts, stats=dict(stats),
             length_buckets={f"{s}:{n}":v for (s,n),v in sorted(hist.items())},
             selection=dict(minimum_tokens=args.minimum_tokens, maximum_tokens=args.maximum_tokens,
                            minimum_edu_score=args.minimum_edu_score, language="full_doc_lid=eng_Latn, score>=0.95",
                            truncated_documents=False, document_deduplication="sha256(text)",
                            split="sha256(text) first64bits mod100 ==0 held-out; before windowing",
                            window="one contiguous hash-selected window per document; no packing",
                            upstream_token_prefilter=args.minimum_tokens//2),
             builder_sha256=sha256(__file__))
    m["manifest_sha256"] = canonical(m)
    (args.output_dir / "repeat.manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m), flush=True)

if __name__ == "__main__":
    main()
