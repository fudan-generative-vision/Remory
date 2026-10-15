"""Shuffle fixed length-similar batches, preserving their exact membership."""
from __future__ import annotations

import hashlib
import json
import random


def global_batches(order, start, threshold, minimum_responses=4):
    """Original batching rule: whole responses, token threshold, merge short tail."""
    while start < len(order):
        end, tokens = start, 0
        while end < len(order) and (tokens < threshold or end - start < minimum_responses):
            tokens += order[end][4]
            end += 1
        if 0 < len(order) - end < minimum_responses:
            tokens += sum(row[4] for row in order[end:])
            end = len(order)
        yield start, end, tokens
        start = end


def epoch_batch_plan(order, *, threshold, epoch, seed):
    """Use an independent RNG; training/model RNG streams are never changed.

    Positions/cursors refer to the concatenation of the shuffled original batches.
    Batches MUST NOT be reconstructed from that concatenation: the merged tail
    can move anywhere in the epoch and must retain its original membership.
    """
    if not order or threshold < 1 or epoch < 0:
        raise ValueError("invalid epoch batch plan")
    base = list(global_batches(order, 0, threshold))
    permutation = list(range(len(base)))
    random.Random(seed + epoch).shuffle(permutation)
    epoch_order, batches = [], []
    for batch_id in permutation:
        begin, end, tokens = base[batch_id]
        start = len(epoch_order)
        epoch_order.extend(order[begin:end])
        batches.append((start, len(epoch_order), tokens, batch_id))
    receipt = dict(epoch=epoch, seed=seed + epoch, batches=len(batches),
                   responses=len(epoch_order), target_tokens=sum(b[2] for b in batches),
                   base_batch_ids=permutation,
                   permutation_sha256=hashlib.sha256(json.dumps(permutation, separators=(",", ":")).encode()).hexdigest())
    return epoch_order, batches, receipt


def resume_batch_index(batches, response_offset):
    """Reject mid-batch cursors rather than silently changing batch membership."""
    for index, batch in enumerate(batches):
        if batch[0] == response_offset:
            return index
    raise ValueError(f"cursor {response_offset} is not an epoch batch boundary")
