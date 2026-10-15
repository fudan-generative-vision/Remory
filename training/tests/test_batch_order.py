import random

import pytest

from context_compression.stage2_batch_order import (
    epoch_batch_plan, global_batches, resume_batch_index,
)


def test_epoch_shuffle_preserves_membership_and_resume():
    # A six-response tail must stay intact even when shuffled into the middle.
    order = [(i, 0, 100 + i, 90 + i, 10) for i in range(18)]
    base = list(global_batches(order, 0, 40))
    random.seed(7)
    state = random.getstate()
    shuffled, batches, receipt = epoch_batch_plan(order, threshold=40, epoch=1, seed=20260917)
    assert random.getstate() == state
    assert sorted(shuffled) == order
    assert receipt['base_batch_ids'] != list(range(len(base)))
    for index, (start, end, tokens, original_id) in enumerate(batches):
        old_start, old_end, old_tokens = base[original_id]
        assert shuffled[start:end] == order[old_start:old_end]
        assert tokens == old_tokens
        assert resume_batch_index(batches, start) == index
    again = epoch_batch_plan(order, threshold=40, epoch=1, seed=20260917)
    assert (shuffled, batches, receipt) == again
    with pytest.raises(ValueError, match='not an epoch batch boundary'):
        resume_batch_index(batches, 1)
