"""Pure chronological, same-depth 1024 -> 64 budget-driven pyramid planning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    index: int
    start: int
    stop: int
    depth: int
    children: tuple[int, ...] = ()


@dataclass(frozen=True)
class PyramidPlan:
    source_tokens: int
    nodes: tuple[Node, ...]
    frontier: tuple[int, ...]
    block: int
    ratio: int
    budget: int

    @property
    def slots_per_node(self):
        return self.block // self.ratio

    @property
    def soft_tokens(self):
        return len(self.frontier) * self.slots_per_node

    @property
    def max_depth(self):
        return max((node.depth for node in self.nodes), default=0)


def plan_pyramid(source_tokens, *, budget=4096, block=1024, ratio=16, max_depth=8,
                 leaf_input_depths=None):
    """All raw spans survive. Only adjacent, equal-depth groups of ratio nodes merge.

    A partial last raw block receives a virtual padded 64-slot allocation, NOT
    synthetic raw input features. Parents always consume 16*64 physical slots.
    Failure to meet a strict budget raises rather than dropping a suffix/history.
    max_depth=None removes the logical recursion ceiling for serving. Embedding
    indices are a separate checkpoint constraint handled by the serving client.
    """
    geometry = (source_tokens, budget, block, ratio)
    if max_depth is not None:
        geometry += (max_depth,)
    for value in geometry:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("pyramid arguments must be positive integers")
    if ratio < 2 or block % ratio or budget < block // ratio:
        raise ValueError("invalid pyramid block/ratio/budget")
    leaf_count = (source_tokens + block - 1) // block
    depths = [0] * leaf_count if leaf_input_depths is None else list(leaf_input_depths)
    if len(depths) != leaf_count or any(
        isinstance(d, bool) or not isinstance(d, int) or d < 0
        or (max_depth is not None and d >= max_depth)
        for d in depths
    ):
        raise ValueError("leaf input depths must be nonnegative integers below any configured depth ceiling")
    nodes = [Node(i, start, min(source_tokens, start + block), depths[i] + 1)
             for i, start in enumerate(range(0, source_tokens, block))]
    frontier = list(range(len(nodes)))
    while len(frontier) * (block // ratio) > budget:
        for start in range(len(frontier) - ratio + 1):
            group = tuple(frontier[start:start + ratio])
            first = nodes[group[0]]
            if max_depth is not None and first.depth >= max_depth:
                continue
            if any(nodes[k].depth != first.depth for k in group):
                continue
            if any(nodes[a].stop != nodes[b].start for a, b in zip(group, group[1:])):
                raise AssertionError("pyramid lost chronological adjacency")
            parent = Node(len(nodes), first.start, nodes[group[-1]].stop, first.depth + 1, group)
            nodes.append(parent)
            frontier[start:start + ratio] = [parent.index]
            break
        else:
            raise ValueError(f"cannot satisfy {budget}-slot budget with same-depth merges; "
                             f"need at least {len(frontier)*(block//ratio)} in this frontier; "
                             "increase budget (no history was truncated)")
    final = [nodes[k] for k in frontier]
    if final[0].start != 0 or final[-1].stop != source_tokens or any(
        a.stop != b.start for a, b in zip(final, final[1:])
    ):
        raise AssertionError("incomplete pyramid coverage")
    return PyramidPlan(source_tokens, tuple(nodes), tuple(frontier), block, ratio, budget)
