import copy
import json
import sqlite3
import zlib

import pytest

from context_compression.qwen38_residual_curriculum import (
    build_context_order, example_lengths, load_context_order, load_ordered_response,
    validate_response_cursor,
)
from context_compression.qwen38_residual_data import identity, residual_events
from context_compression.train_qwen38_mmfine_joint import QueueCursor


def make_database():
    # Sorting trajectories cannot achieve the global response order 0:0, 1:0, 0:1.
    examples = [
        {"key": "coding:0", "corpus": "coding", "prefix_ids": [1], "segments": [
            {"context_ids": [2]*3, "opening_ids": [3], "response_ids": [4]*300},
            {"context_ids": [5]*2, "opening_ids": [3], "response_ids": [6]}]},
        {"key": "xhigh:1", "corpus": "xhigh", "prefix_ids": [1], "segments": [
            {"context_ids": [2]*10, "opening_ids": [3], "response_ids": [7]*8}]},
        {"key": "coding:2", "corpus": "coding", "prefix_ids": [1]*8, "segments": [
            {"context_ids": [2]*3, "opening_ids": [3], "response_ids": [8]*2}]},
    ]
    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE examples (ordinal INTEGER PRIMARY KEY, payload BLOB)")
    database.executemany("INSERT INTO examples VALUES (?, ?)",
        [(i, zlib.compress(json.dumps(ex).encode())) for i, ex in enumerate(examples)])
    return database, examples


def write_plan(tmp_path):
    database, examples = make_database()
    plan = build_context_order(database, database_sha256="fixed-data")
    path = tmp_path / "order.json"
    path.write_text(json.dumps(plan))
    manifest = {"database_sha256": "fixed-data", "stats": {"all": {
        "trajectories": 3, "responses": 4, "target_tokens": 311}}}
    return database, examples, plan, path, manifest


def test_global_response_order_stable_ties_and_complete_coverage(tmp_path):
    db, examples, plan, path, manifest = write_plan(tmp_path)
    order, receipt = load_context_order(path, dataset_manifest=manifest)
    assert [(r[0], r[1]) for r in order] == [(0, 0), (1, 0), (2, 0), (0, 1)]
    assert [r[2] for r in order] == [4, 11, 11, 307]
    assert receipt["target_tokens"] == 311 and receipt["selected_responses"] == 4
    assert plan == build_context_order(db, database_sha256="fixed-data")
    for row in order:
        event, corpus, last = load_ordered_response(db, row)
        assert event == list(residual_events(examples[row[0]]))[row[1]]
        assert corpus == examples[row[0]]["corpus"]
        assert last == (row[1] == len(examples[row[0]]["segments"])-1)


def test_current_answer_does_not_change_its_context_length():
    db, examples = make_database()
    old = list(example_lengths(examples[0], 0))
    changed = copy.deepcopy(examples[0])
    changed["segments"][0]["response_ids"] *= 2
    new = list(example_lengths(changed, 0))
    assert old[0][2:4] == new[0][2:4]
    assert new[0][4] == old[0][4]*2
    assert new[1][2] == old[1][2]+300  # Earlier answers correctly belong to later H.


def test_canary_subset_is_still_globally_sorted(tmp_path):
    _, _, _, path, manifest = write_plan(tmp_path)
    order, receipt = load_context_order(path, dataset_manifest=manifest, canary_ordinals=[2, 0])
    assert [(r[0], r[1]) for r in order] == [(0, 0), (2, 0), (0, 1)]
    assert receipt["selected_responses"] == 3
    with pytest.raises(ValueError, match="missing"):
        load_context_order(path, dataset_manifest=manifest, canary_ordinals=[7])


@pytest.mark.parametrize("damage", ["hash", "order", "duplicate", "coverage", "database"])
def test_invalid_plan_fails_closed(tmp_path, damage):
    _, _, plan, path, manifest = write_plan(tmp_path)
    if damage == "database":
        manifest["database_sha256"] = "other"
    elif damage == "hash":
        plan["plan_sha256"] = "bad"
    else:
        if damage == "order":
            plan["order"].reverse()
        elif damage == "duplicate":
            plan["order"][2] = plan["order"][1]
        else:
            plan["order"].pop()
        plan["plan_sha256"] = identity({k:v for k,v in plan.items() if k != "plan_sha256"})
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_context_order(path, dataset_manifest=manifest)


def test_resume_inside_interleaved_trajectory_and_epoch_boundaries(tmp_path):
    db, _, _, path, manifest = write_plan(tmp_path)
    order, _ = load_context_order(path, dataset_manifest=manifest)
    cursor = QueueCursor(0, 0, 2, 0)
    validate_response_cursor(cursor, epochs=2, responses=len(order))
    before = [load_ordered_response(db, row)[0] for row in order[:cursor.event_index]]
    reloaded, _ = load_context_order(path, dataset_manifest=manifest)
    after = [load_ordered_response(db, row)[0] for row in reloaded[cursor.event_index:]]
    assert [(e.key, e.turn) for e in before+after] == [
        (load_ordered_response(db, row)[0].key, row[1]) for row in order]
    # The next epoch restarts the same sorted order, not a reshuffle.
    validate_response_cursor(QueueCursor(1, 0, 0, 0), epochs=2, responses=len(order))
    validate_response_cursor(QueueCursor(2, 0, 0, 0), epochs=2, responses=len(order))
    for invalid in [QueueCursor(0, 0, 4, 0), QueueCursor(2, 0, 1, 0),
                    QueueCursor(0, 1, 0, 0), QueueCursor(3, 0, 0, 0)]:
        with pytest.raises(ValueError, match="cursor"):
            validate_response_cursor(invalid, epochs=2, responses=len(order))


def test_wrong_runtime_lengths_rejected(tmp_path):
    db, _, plan, _, _ = write_plan(tmp_path)
    row = list(plan["order"][0])
    row[2] += 1
    with pytest.raises(ValueError, match="lengths differ"):
        load_ordered_response(db, row)
