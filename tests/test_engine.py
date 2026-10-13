from dataclasses import replace
import sqlite3

import numpy as np
import pytest

from remory import MemoryStore, Remory
from remory.pyramid import plan_pyramid
from conftest import compact


def test_pyramid_covers_every_real_token():
    for count in range(1, 130):
        plan = plan_pyramid(count, budget=16, block=8, ratio=2, max_depth=None)
        nodes = [plan.nodes[k] for k in plan.frontier]
        assert nodes[0].start == 0 and nodes[-1].stop == count
        assert all(a.stop == b.start for a, b in zip(nodes, nodes[1:]))
        assert plan.soft_tokens <= 16
    with pytest.raises(ValueError, match="no history was truncated"):
        plan_pyramid(113, budget=12, block=8, ratio=2, max_depth=None)
    assert plan_pyramid(1, block=8, ratio=2, budget=4).soft_tokens == 4


def test_recursive_compaction_reconditions_old_memory(runtime):
    first = compact(runtime)
    assert first.embeddings.shape == (8, 3)
    runtime.generate(owner="session-a", handle=first.handle, continuation_ids=[70, 71], max_new_tokens=5)
    restored = runtime.backend.calls[-1][0]
    assert restored.input_ids[-2:] == (70, 71)
    assert [restored.input_ids[p] for p in restored.memory_positions] == [2] * 8
    old_rows = first.embeddings.copy()
    second = runtime.compact(owner="session-a", prefix_ids=[1, 3], history_ids=[70, 71],
                             summary_ids=[60], previous=first.handle)
    source, kwargs = next((s, k) for s, k in runtime.backend.calls if k.get("operation") == "recursive_leaves")
    assert source.input_ids == restored.input_ids
    np.testing.assert_array_equal(source.memory, old_rows)
    assert max(kwargs["block_depths"]) > 0
    assert second.receipt["source_memory_slots"] == 8
    assert second.summary_ids == (60,)
    assert second.handle != first.handle
    np.testing.assert_array_equal(runtime.store.get("session-a", first.handle).embeddings, old_rows)


def test_failure_does_not_commit_or_delete_history(runtime):
    first = compact(runtime)
    runtime.backend.fail = "recursive_leaves"
    with pytest.raises(RuntimeError):
        runtime.compact(owner="session-a", prefix_ids=[1, 3], history_ids=[80],
                        summary_ids=[90], previous=first.handle)
    assert runtime.store.get("session-a", first.handle).summary_ids == (50, 51)
    with sqlite3.connect(runtime.store.path) as db:
        assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


def test_store_survives_restart_and_owner_mismatch(runtime):
    memory = compact(runtime)
    restarted = Remory(runtime.backend, MemoryStore(runtime.store.path))
    restarted.generate(owner="session-a", handle=memory.handle, max_new_tokens=5)
    assert runtime.backend.calls[-1][0].memory.shape == (8, 3)
    with pytest.raises(KeyError):
        restarted.generate(owner="session-b", handle=memory.handle, max_new_tokens=5)
    runtime.contract = replace(runtime.contract, identity="other-checkpoint")
    with pytest.raises(ValueError, match="different model"):
        runtime.generate(owner="session-a", handle=memory.handle, max_new_tokens=5)


def test_reject_changed_prefix_and_context_overflow(runtime):
    memory = compact(runtime)
    with pytest.raises(ValueError, match="prefix changed"):
        runtime.compact(owner="session-a", prefix_ids=[3], history_ids=[6], summary_ids=[7], previous=memory.handle)
    runtime.backend.context_limit = 4
    before = len(runtime.backend.calls)
    with pytest.raises(ValueError, match="not truncated"):
        compact(runtime)
    assert len(runtime.backend.calls) == before


def test_generate_passes_actual_embedding_overrides(runtime):
    memory = compact(runtime)
    output = runtime.generate(owner="session-a", handle=memory.handle, continuation_ids=[99], max_new_tokens=5)
    source, kwargs = runtime.backend.calls[-1]
    np.testing.assert_array_equal(source.memory, memory.embeddings)
    assert source.input_ids[-1] == 99
    assert output.text == "ok" and kwargs["max_new_tokens"] == 5
    with pytest.raises(ValueError):
        runtime.generate(owner="session-a", handle=memory.handle, input_ids=[1], max_new_tokens=5)


@pytest.mark.parametrize("bad", [[True], [-1], [1.5], [], "tokens"])
def test_bad_tokens_fail_before_backend(runtime, bad):
    with pytest.raises(ValueError):
        runtime.compact(owner="a", prefix_ids=bad, history_ids=[5], summary_ids=[6])
    assert not runtime.backend.calls


def test_nonfinite_rows_never_commit(runtime):
    runtime.backend.encode = lambda *args, **kwargs: np.full((2, 3), np.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        compact(runtime)
    with sqlite3.connect(runtime.store.path) as db:
        assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 0


def test_fold_reencodes_child_memory_and_preserves_owner_isolation(runtime):
    parent = compact(runtime)
    child = runtime.compact(owner='session-a', prefix_ids=[7, 8], history_ids=[30, 31],
                            summary_ids=[90])
    old_child = child.embeddings.copy()
    folded = runtime.compact(owner='session-a', prefix_ids=[1, 3], history_ids=[80],
        summary_ids=[60], previous=parent.handle, branches=[child.handle])
    assert folded.receipt['branches'] == [child.handle]
    assert folded.receipt['source_memory_slots'] == len(parent.embeddings) + len(child.embeddings)
    np.testing.assert_array_equal(runtime.store.get('session-a', child.handle).embeddings, old_child)
    runtime.generate(owner='session-a', handle=folded.handle, continuation_ids=[88], max_new_tokens=4)
    with pytest.raises(KeyError):
        runtime.compact(owner='session-b', prefix_ids=[1], history_ids=[2], summary_ids=[3],
                        branches=[child.handle])
    with pytest.raises(ValueError, match='twice'):
        runtime.compact(owner='session-a', prefix_ids=[1, 3], history_ids=[80], summary_ids=[60],
                        previous=parent.handle, branches=[child.handle, child.handle])


def test_compact_rejects_unrestorable_summary_before_encoding(runtime):
    runtime.backend.context_limit = 32
    with pytest.raises(ValueError, match='not truncated'):
        runtime.compact(owner='session-a', prefix_ids=[1, 3], history_ids=[20], summary_ids=[50] * 24)
    assert not runtime.backend.calls
