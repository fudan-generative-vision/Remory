from __future__ import annotations

from dataclasses import asdict
import numpy as np

from .pyramid import plan_pyramid
from .store import Memory, MemoryStore
from .types import Backend, Prepared, digest, token_ids


class Remory:
    def __init__(self, backend: Backend, store: MemoryStore):
        self.backend, self.store = backend, store
        self.contract = backend.contract

    def _check_length(self, size: int):
        if size > self.backend.context_limit:
            raise ValueError(f"{size} tokens exceed backend limit {self.backend.context_limit}; "
                             "history was not truncated")

    def _check_memory(self, memory: Memory):
        if memory.identity != self.contract.identity:
            raise ValueError("memory belongs to a different model, checkpoint, or contract")
        self._matrix(memory.embeddings, len(memory.slot_depths))

    def _matrix(self, values, rows):
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (rows, self.contract.hidden_size) or not np.isfinite(result).all():
            raise ValueError("backend returned invalid memory shape or nonfinite values")
        return result

    def _prepare(self, *, owner: str, handle: str, continuation_ids=()) -> Prepared:
        """Restore P + summary + soft memory + the exact native continuation IDs.

        The placeholder IDs MUST be accompanied by the embedding overrides.
        Sending just input_ids to an ordinary model API discards residual memory.
        """
        m = self.store.get(owner, handle)
        self._check_memory(m)
        tail = token_ids(continuation_ids, "continuation_ids", empty=True)
        c = self.contract
        before = m.prefix_ids + c.summary_before + m.summary_ids + c.summary_after + c.memory_before
        positions = tuple(range(len(before), len(before) + m.embeddings.shape[0]))
        ids = before + (c.placeholder_id,) * len(positions) + c.memory_after + tail
        self._check_length(len(ids))
        return Prepared(ids, positions, m.embeddings, m.slot_depths)

    def compact(self, *, owner: str, prefix_ids, history_ids, summary_ids,
                previous: str | None = None, branches=()) -> Memory:
        """Commit a new immutable checkpoint after all encode stages succeed.

        Initially history_ids is the removed raw span. With previous set, it is
        ONLY the new span since that checkpoint (including any previously kept
        tail now being removed). The old summary + soft rows are re-encoded,
        conditioned on the NEW summary. Original prefix IDs must stay identical.
        Retained recent messages belong in the subsequent generate continuation.
        """
        self.store._key(owner)
        prefix = token_ids(prefix_ids, "prefix_ids")
        history = token_ids(history_ids, "history_ids", empty=previous is not None or bool(branches))
        summary_ids = token_ids(summary_ids, "summary_ids")
        c = self.contract
        if previous:
            old = self.store.get(owner, previous)
            self._check_memory(old)
            if old.prefix_ids != prefix:
                raise ValueError("original system/tools/task prefix changed across compaction")
            source = self._prepare(owner=owner, handle=previous, continuation_ids=history)
        else:
            source = Prepared(prefix + history)
        if not isinstance(branches, (list, tuple)) or len(branches) > 16:
            raise ValueError("branches must contain at most 16 memory handles")
        if len(set(branches)) != len(branches) or previous in branches:
            raise ValueError("branch memory must not be included twice")
        # Child checkpoints remain immutable. Only owner-bound memories from the
        # same actor/encoder may join the parent's next residual frontier.
        for handle in branches:
            child = self.store.get(owner, handle)
            self._check_memory(child)
            lead = source.input_ids + c.summary_before + child.summary_ids + c.summary_after + c.memory_before
            positions = tuple(range(len(lead), len(lead) + len(child.embeddings)))
            memory = (np.concatenate([source.memory, child.embeddings])
                      if source.memory is not None else child.embeddings)
            source = Prepared(lead + (c.placeholder_id,) * len(positions) + c.memory_after,
                source.memory_positions + positions, memory, source.slot_depths + child.slot_depths)
        summary_start = len(prefix) + len(c.summary_before)
        summary_source = Prepared(prefix + c.summary_before + summary_ids + c.summary_after)
        self._check_length(max(len(source.input_ids), len(summary_source.input_ids),
                               len(prefix) + c.block_tokens))
        source_tokens = len(source.input_ids) - len(prefix)
        depths = [0] * ((source_tokens + c.block_tokens - 1) // c.block_tokens)
        for pos, depth in zip(source.memory_positions, source.slot_depths):
            index = (pos - len(prefix)) // c.block_tokens
            depths[index] = max(depths[index], depth)
        plan = plan_pyramid(source_tokens, budget=c.memory_budget, block=c.block_tokens,
                            ratio=c.ratio, max_depth=None, leaf_input_depths=depths)
        self._check_length(len(prefix) + len(c.summary_before) + len(summary_ids)
                           + len(c.summary_after) + len(c.memory_before)
                           + plan.soft_tokens + len(c.memory_after))
        summary = self._matrix(self.backend.encode(
            summary_source, start=summary_start, end=summary_start + len(summary_ids),
            operation="summary"), len(summary_ids))
        leaves = [n for n in plan.nodes if not n.children]
        leaf_rows = self._matrix(self.backend.encode(
            source, start=len(prefix), end=len(source.input_ids),
            operation="recursive_leaves" if source.memory_positions else "leaves", summary=summary,
            block_depths=tuple(min(d, c.max_depth - 1) for d in depths) if source.memory_positions else ()),
            len(leaves) * plan.slots_per_node)
        outputs = {n.index: leaf_rows[i * plan.slots_per_node:(i + 1) * plan.slots_per_node]
                   for i, n in enumerate(leaves)}
        for node in plan.nodes:
            if not node.children:
                continue
            children = np.concatenate([outputs.pop(k) for k in node.children])
            parent_source = Prepared(prefix + (c.placeholder_id,) * c.block_tokens,
                                     tuple(range(len(prefix), len(prefix) + c.block_tokens)), children)
            outputs[node.index] = self._matrix(self.backend.encode(
                parent_source, start=len(prefix), end=len(parent_source.input_ids), operation="parent",
                summary=summary, input_depth=min(node.depth - 1, c.max_depth - 1)), plan.slots_per_node)
        memory = np.concatenate([outputs[k] for k in plan.frontier])
        slot_depths = tuple(plan.nodes[k].depth for k in plan.frontier
                            for _ in range(plan.slots_per_node))
        receipt = dict(schema="remory.compaction.v1", source_tokens=source_tokens,
                       summary_tokens=len(summary_ids), soft_tokens=plan.soft_tokens,
                       prefix_sha256=digest(prefix), summary_sha256=digest(summary_ids),
                       history_sha256=digest(source.input_ids), previous=previous,
                       branches=list(branches),
                       source_memory_slots=len(source.memory_positions),
                       block_input_depths=depths, logical_max_depth=plan.max_depth,
                       embedding_depth_saturated=any(n.depth > c.max_depth for n in plan.nodes),
                       frontier=list(plan.frontier), nodes=[asdict(n) for n in plan.nodes])
        return self.store.put(owner=owner, identity=c.identity, prefix_ids=prefix,
                              summary_ids=summary_ids, embeddings=memory,
                              slot_depths=slot_depths, receipt=receipt)

    def generate(self, *, owner: str, max_new_tokens: int, handle: str | None = None,
                 continuation_ids=(), input_ids=None, sampling=None):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if handle:
            if input_ids is not None:
                raise ValueError("use continuation_ids with a handle, not full input_ids")
            source = self._prepare(owner=owner, handle=handle, continuation_ids=continuation_ids)
        else:
            self.store._key(owner)
            if continuation_ids:
                raise ValueError("continuation_ids requires a memory handle")
            source = Prepared(token_ids(input_ids, "input_ids"))
        self._check_length(len(source.input_ids) + max_new_tokens)
        return self.backend.generate(source, max_new_tokens=max_new_tokens, sampling=sampling)
