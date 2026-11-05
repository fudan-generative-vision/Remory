"""Stable all-response curriculum: original context length ascending every epoch."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

from context_compression.qwen38_allturn_data import SCHEMA as DATA_SCHEMA, digest, load_example, open_database
from context_compression.qwen38_residual_data import ResidualEvent, identity

SCHEMA = "qwen38_residual_context_ascending_response_order_v1"
COLUMNS = ["ordinal", "turn", "context_tokens", "history_tokens", "target_tokens"]
POLICY = {"unit": "response", "coordinate": "native_prefix_plus_complete_history_excluding_opening_and_current_answer",
          "sort": ["context_tokens", "ordinal", "turn"], "each_epoch": "same_ascending_order_no_shuffle"}


def example_lengths(example, ordinal):
    history = 0
    for turn, segment in enumerate(example["segments"]):
        history += len(segment["context_ids"])
        target = len(segment["response_ids"])
        if not history or not target or not segment["opening_ids"]:
            raise ValueError("empty history/opening/answer in curriculum")
        yield [ordinal, turn, len(example["prefix_ids"]) + history, history, target]
        history += len(segment["opening_ids"]) + target


def build_context_order(database, *, database_sha256):
    order = []
    trajectories = 0
    for ordinal, payload in database.execute("SELECT ordinal, payload FROM examples ORDER BY ordinal"):
        order.extend(example_lengths(json.loads(zlib.decompress(payload)), ordinal))
        trajectories += 1
    order.sort(key=lambda row: (row[2], row[0], row[1]))
    if not order:
        raise ValueError("empty context curriculum")
    plan = {"schema": SCHEMA, "database_sha256": database_sha256, "policy": POLICY,
            "columns": COLUMNS, "trajectories": trajectories, "responses": len(order),
            "target_tokens": sum(row[4] for row in order),
            "min_context_tokens": order[0][2], "max_context_tokens": order[-1][2],
            "order": order}
    return {**plan, "plan_sha256": identity(plan)}


def load_context_order(path, *, dataset_manifest, canary_ordinals=None):
    plan = json.loads(Path(path).read_text())
    content = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan["schema"] != SCHEMA or plan["policy"] != POLICY or plan["columns"] != COLUMNS
            or plan["database_sha256"] != dataset_manifest["database_sha256"]
            or plan["plan_sha256"] != identity(content)):
        raise ValueError("context-order plan identity/contract mismatch")
    order = plan["order"]
    if not order or any(len(r) != len(COLUMNS) or any(type(x) is not int or x < 0 for x in r)
                        or r[2] < r[3] or r[3] == 0 or r[4] == 0 for r in order):
        raise ValueError("invalid context-order rows")
    if any((a[2], a[0], a[1]) > (b[2], b[0], b[1]) for a, b in zip(order, order[1:])):
        raise ValueError("context order is not ascending")
    expected = tuple(sum(s[k] for s in dataset_manifest["stats"].values())
                     for k in ("trajectories", "responses", "target_tokens"))
    observed = len({r[0] for r in order}), len(order), sum(r[4] for r in order)
    if (observed != expected or observed != (plan["trajectories"], plan["responses"], plan["target_tokens"])
            or len({(r[0], r[1]) for r in order}) != len(order)
            or (order[0][2], order[-1][2]) != (plan["min_context_tokens"], plan["max_context_tokens"])):
        raise ValueError("context-order coverage mismatch")
    if canary_ordinals:
        selected = set(canary_ordinals)
        order = [row for row in order if row[0] in selected]
        if {r[0] for r in order} != selected:
            raise ValueError("canary ordinals missing from context order")
    receipt = {k: v for k, v in plan.items() if k != "order"}
    receipt.update(selected_responses=len(order), selected_min_context_tokens=order[0][2],
                   selected_max_context_tokens=order[-1][2])
    return order, receipt


def load_ordered_response(database, row):
    ordinal, turn, context_tokens, history_tokens, target_tokens = row
    example = load_example(database, ordinal)
    if not 0 <= turn < len(example["segments"]):
        raise ValueError("context-order turn is outside trajectory")
    history = []
    for previous in example["segments"][:turn]:
        history.extend(previous["context_ids"])
        history.extend(previous["opening_ids"])
        history.extend(previous["response_ids"])
    segment = example["segments"][turn]
    history.extend(segment["context_ids"])
    event = ResidualEvent(example["key"], turn, tuple(example["prefix_ids"]), tuple(history),
                          tuple(segment["opening_ids"]), tuple(segment["response_ids"]),
                          (tuple(segment["student_opening_ids"])
                           if "student_opening_ids" in segment else None),
                          tuple(segment.get("summary_closing_ids", ())),
                          segment.get("target_offset", 0))
    if (len(event.prefix_ids) + len(history), len(history), len(event.target_ids)) != (
            context_tokens, history_tokens, target_tokens):
        raise ValueError("context-order lengths differ from exact causal response")
    return event, example["corpus"], turn + 1 == len(example["segments"])


def validate_response_cursor(cursor, *, epochs, responses):
    if (cursor.shard_index or cursor.chunk_index or not 0 <= cursor.epoch <= epochs
            or responses <= 0 or not 0 <= cursor.event_index < responses
            or (cursor.epoch == epochs and cursor.event_index != 0)):
        raise ValueError("invalid context-sorted response cursor")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("context-order plan already exists; refusing to overwrite")
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != DATA_SCHEMA or digest(Path(manifest["database"])) != manifest["database_sha256"]:
        raise ValueError("allturn database identity mismatch")
    database = open_database(Path(manifest["database"]))
    try:
        plan = build_context_order(database, database_sha256=manifest["database_sha256"])
    finally:
        database.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(plan, handle, separators=(",", ":"))
    _, receipt = load_context_order(args.output, dataset_manifest=manifest)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
