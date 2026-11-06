from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class DDTreeBuildResult:
    """Fixed-width DDTree representation for a batch.

    Node 0 is always the already-verified root/bonus token. The remaining
    ``tree_budget`` rows either contain proposal-trie nodes or unreachable
    padding. ``build_ddtree_tree`` selects proposal nodes by best-first search
    over a factorized DFlash block distribution. ``build_candidate_union_tree``
    instead inserts explicit greedy chains and pads their prefix union to a
    fixed target-verification width.
    """

    node_token_ids: torch.Tensor  # [bs, tree_budget], excludes root
    node_depths: torch.Tensor  # [bs, tree_budget], root-relative depths in [1, L]
    parents: torch.Tensor  # [bs, tree_budget + 1], root parent = -1
    visibility: torch.Tensor  # [bs, N, N], ancestor/self visibility
    retrieve_next_token: torch.Tensor  # [bs, N], first-child pointer
    retrieve_next_sibling: torch.Tensor  # [bs, N], next-sibling pointer
    child_maps: List[Dict[int, Dict[int, int]]]
    # Optional diagnostics for explicit candidate-union trees. Counts exclude
    # root node 0. ``valid_node_mask`` includes the root and excludes padding.
    actual_node_counts: Optional[torch.Tensor] = None  # [bs]
    unique_chain_counts: Optional[torch.Tensor] = None  # [bs]
    valid_node_mask: Optional[torch.Tensor] = None  # [bs, N]

    @property
    def max_tree_nodes(self) -> int:
        return int(self.parents.shape[1])


def _build_single_tree(
    top_log_probs: torch.Tensor,
    top_token_ids: torch.Tensor,
    tree_budget: int,
    active_k: torch.Tensor | None = None,
) -> Tuple[List[int], List[int], List[int], Dict[int, Dict[int, int]]]:
    """Build the highest-scoring prefix-closed node set with a heap.

    ``top_log_probs`` and ``top_token_ids`` have shape [depth_limit, topk].
    A node at depth d receives the factorized prefix score
    ``sum_{j < d} log q_j(token_j)``.
    """

    if tree_budget <= 0:
        return [], [], [-1], {0: {}}

    depth_limit, topk = top_log_probs.shape
    if depth_limit <= 0 or topk <= 0:
        raise ValueError(
            f"DDTree requires non-empty draft distributions, got shape={tuple(top_log_probs.shape)}."
        )

    logp = top_log_probs.to(device="cpu", dtype=torch.float64)
    token_ids = top_token_ids.to(device="cpu", dtype=torch.long)
    if active_k is None:
        active = torch.full((depth_limit,), topk, dtype=torch.long)
    else:
        active = active_k.to(device="cpu", dtype=torch.long)
        if active.shape != (depth_limit,):
            raise ValueError(
                f"DDTree active_k must have shape {(depth_limit,)}, got {tuple(active.shape)}."
            )
        active = active.clamp(min=1, max=topk)

    # Heap item:
    #   (-prefix_score, insertion_order, parent_node, depth, rank, prefix_score)
    # insertion_order gives deterministic tie-breaking.
    insertion_order = 0
    first_score = float(logp[0, 0].item())
    heap: List[Tuple[float, int, int, int, int, float]] = [
        (-first_score, insertion_order, 0, 1, 0, first_score)
    ]

    node_tokens: List[int] = []
    node_depths: List[int] = []
    parents: List[int] = [-1]
    child_maps: Dict[int, Dict[int, int]] = {0: {}}

    while heap and len(node_tokens) < tree_budget:
        _, _, parent_idx, depth, rank, prefix_score = heapq.heappop(heap)
        token_id = int(token_ids[depth - 1, rank].item())
        node_idx = len(node_tokens) + 1

        node_tokens.append(token_id)
        node_depths.append(depth)
        parents.append(parent_idx)
        child_maps.setdefault(parent_idx, {})[token_id] = node_idx
        child_maps.setdefault(node_idx, {})

        # The next sibling replaces only the final token in this prefix.
        # active_k keeps tensor shapes static while allowing a dynamic number
        # of candidates per depth.
        if rank + 1 < int(active[depth - 1].item()):
            sibling_score = (
                prefix_score
                - float(logp[depth - 1, rank].item())
                + float(logp[depth - 1, rank + 1].item())
            )
            insertion_order += 1
            heapq.heappush(
                heap,
                (
                    -sibling_score,
                    insertion_order,
                    parent_idx,
                    depth,
                    rank + 1,
                    sibling_score,
                ),
            )

        # The best child extends this exact prefix by one position.
        if depth < depth_limit:
            child_score = prefix_score + float(logp[depth, 0].item())
            insertion_order += 1
            heapq.heappush(
                heap,
                (
                    -child_score,
                    insertion_order,
                    node_idx,
                    depth + 1,
                    0,
                    child_score,
                ),
            )

    if len(node_tokens) != tree_budget:
        raise RuntimeError(
            "DDTree heap exhausted before reaching the requested budget. "
            f"built={len(node_tokens)}, budget={tree_budget}, depth_limit={depth_limit}, topk={topk}."
        )

    return node_tokens, node_depths, parents, child_maps


def _compile_topology(
    parents: Sequence[int],
    child_map: Dict[int, Dict[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile ancestor visibility plus first-child/next-sibling pointers."""

    num_nodes = len(parents)
    visibility = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    visibility[0, 0] = True

    for node_idx in range(1, num_nodes):
        parent_idx = int(parents[node_idx])
        if parent_idx < 0 or parent_idx >= node_idx:
            raise RuntimeError(
                f"Invalid DDTree parent: node={node_idx}, parent={parent_idx}."
            )
        visibility[node_idx].copy_(visibility[parent_idx])
        visibility[node_idx, node_idx] = True

    retrieve_next_token = torch.full((num_nodes,), -1, dtype=torch.long)
    retrieve_next_sibling = torch.full((num_nodes,), -1, dtype=torch.long)

    for parent_idx, token_to_child in child_map.items():
        children = list(token_to_child.values())
        if not children:
            continue
        retrieve_next_token[parent_idx] = int(children[0])
        for left, right in zip(children[:-1], children[1:]):
            retrieve_next_sibling[int(left)] = int(right)

    return visibility, retrieve_next_token, retrieve_next_sibling


def _build_single_candidate_union(
    candidate_chains: torch.Tensor,
    *,
    tree_budget: int,
) -> Tuple[
    List[int],
    List[int],
    List[int],
    Dict[int, Dict[int, int]],
    int,
    int,
]:
    """Build and fixed-width pad the prefix union of explicit token chains.

    ``candidate_chains`` has shape ``[M, L]``. New nodes are inserted in stable
    sample-major, then depth-major order. Padding nodes have root as their
    attention parent but are deliberately omitted from the root child map.
    Consequently, they occupy target-verifier rows without being reachable by
    :func:`follow_verified_tree`.
    """

    if candidate_chains.ndim != 2:
        raise ValueError(
            "DDTree candidate chains must have shape [M, L], got "
            f"{tuple(candidate_chains.shape)}."
        )
    num_samples, horizon = map(int, candidate_chains.shape)
    if num_samples <= 0 or horizon <= 0:
        raise ValueError(
            "DDTree candidate union requires positive M and L, got "
            f"M={num_samples}, L={horizon}."
        )
    expected_budget = num_samples * horizon
    if int(tree_budget) != expected_budget:
        raise ValueError(
            "DDTree candidate-union budget must equal M * L so target verify "
            f"width stays fixed: budget={tree_budget}, M={num_samples}, "
            f"L={horizon}, expected={expected_budget}."
        )

    chains = candidate_chains.to(device="cpu", dtype=torch.long)
    node_tokens: List[int] = []
    node_depths: List[int] = []
    parents: List[int] = [-1]
    child_maps: Dict[int, Dict[int, int]] = {0: {}}
    unique_chains = set()

    for sample_idx in range(num_samples):
        chain = tuple(int(token) for token in chains[sample_idx].tolist())
        unique_chains.add(chain)
        parent_idx = 0
        for depth, token_id in enumerate(chain, start=1):
            children = child_maps.setdefault(parent_idx, {})
            node_idx = children.get(token_id)
            if node_idx is None:
                node_idx = len(node_tokens) + 1
                if node_idx > tree_budget:
                    raise RuntimeError(
                        "DDTree candidate-union trie exceeded its M * L budget: "
                        f"node={node_idx}, budget={tree_budget}."
                    )
                node_tokens.append(token_id)
                node_depths.append(depth)
                parents.append(parent_idx)
                children[token_id] = node_idx
                child_maps[node_idx] = {}
            parent_idx = int(node_idx)

    actual_node_count = len(node_tokens)
    if actual_node_count <= 0:
        raise RuntimeError("DDTree candidate-union trie unexpectedly has no nodes.")

    # Use a known-valid proposal token for padding. Padding rows are valid model
    # inputs and attend to root/self, but no real trie node has an edge to them.
    # They are therefore impossible for follow_verified_tree() to accept.
    pad_token_id = int(node_tokens[0])
    while len(node_tokens) < tree_budget:
        padding_node_idx = len(node_tokens) + 1
        node_tokens.append(pad_token_id)
        node_depths.append(1)
        parents.append(0)
        child_maps[padding_node_idx] = {}

    return (
        node_tokens,
        node_depths,
        parents,
        child_maps,
        actual_node_count,
        len(unique_chains),
    )


def build_candidate_union_tree(
    *,
    candidate_token_chains: torch.Tensor,
    tree_budget: int,
    device: torch.device,
) -> DDTreeBuildResult:
    """Compile ``M`` greedy chains per request into one fixed-width trie.

    Args:
        candidate_token_chains: Explicit proposal chains with shape ``[M, bs, L]``.
        tree_budget: Fixed non-root verify width. It must equal ``M * L``.
        device: Destination device for runtime tensors.

    Identical chains and shared prefixes are represented only once. Any rows
    saved by this union are filled with unreachable padding, keeping the target
    verify width exactly ``1 + M * L`` for CUDA-graph compatibility.
    """

    if candidate_token_chains.ndim != 3:
        raise ValueError(
            "DDTree candidate_token_chains must have shape [M, bs, L], got "
            f"{tuple(candidate_token_chains.shape)}."
        )
    num_samples, bs, horizon = map(int, candidate_token_chains.shape)
    if num_samples <= 0 or bs <= 0 or horizon <= 0:
        raise ValueError(
            "DDTree candidate union requires positive M, batch, and L, got "
            f"M={num_samples}, batch={bs}, L={horizon}."
        )
    expected_budget = num_samples * horizon
    if int(tree_budget) != expected_budget:
        raise ValueError(
            "DDTree candidate-union budget must equal M * L so target verify "
            f"width stays fixed: budget={tree_budget}, M={num_samples}, "
            f"L={horizon}, expected={expected_budget}."
        )

    # One batched D2H transfer; Python trie construction then uses CPU values.
    chains_cpu = candidate_token_chains.to(device="cpu", dtype=torch.long)
    max_nodes = int(tree_budget) + 1

    all_tokens: List[List[int]] = []
    all_depths: List[List[int]] = []
    all_parents: List[List[int]] = []
    all_visibility: List[torch.Tensor] = []
    all_next_token: List[torch.Tensor] = []
    all_next_sibling: List[torch.Tensor] = []
    all_child_maps: List[Dict[int, Dict[int, int]]] = []
    actual_node_counts: List[int] = []
    unique_chain_counts: List[int] = []
    valid_node_masks: List[torch.Tensor] = []

    for batch_idx in range(bs):
        (
            tokens,
            depths,
            parents,
            child_map,
            actual_node_count,
            unique_chain_count,
        ) = _build_single_candidate_union(
            chains_cpu[:, batch_idx, :],
            tree_budget=int(tree_budget),
        )
        visibility, next_token, next_sibling = _compile_topology(parents, child_map)
        valid_node_mask = torch.zeros((max_nodes,), dtype=torch.bool)
        valid_node_mask[: actual_node_count + 1] = True

        all_tokens.append(tokens)
        all_depths.append(depths)
        all_parents.append(parents)
        all_visibility.append(visibility)
        all_next_token.append(next_token)
        all_next_sibling.append(next_sibling)
        all_child_maps.append(child_map)
        actual_node_counts.append(actual_node_count)
        unique_chain_counts.append(unique_chain_count)
        valid_node_masks.append(valid_node_mask)

    return DDTreeBuildResult(
        node_token_ids=torch.tensor(
            all_tokens, dtype=torch.long, device=device
        ).view(bs, tree_budget),
        node_depths=torch.tensor(
            all_depths, dtype=torch.long, device=device
        ).view(bs, tree_budget),
        parents=torch.tensor(
            all_parents, dtype=torch.long, device=device
        ).view(bs, max_nodes),
        visibility=torch.stack(all_visibility, dim=0).to(device=device),
        retrieve_next_token=torch.stack(all_next_token, dim=0).to(device=device),
        retrieve_next_sibling=torch.stack(all_next_sibling, dim=0).to(device=device),
        child_maps=all_child_maps,
        actual_node_counts=torch.tensor(
            actual_node_counts, dtype=torch.long, device=device
        ),
        unique_chain_counts=torch.tensor(
            unique_chain_counts, dtype=torch.long, device=device
        ),
        valid_node_mask=torch.stack(valid_node_masks, dim=0).to(device=device),
    )


def build_ddtree_tree(
    *,
    top_log_probs: torch.Tensor,
    top_token_ids: torch.Tensor,
    tree_budget: int,
    device: torch.device,
    active_k: torch.Tensor | None = None,
) -> DDTreeBuildResult:
    """Build a DDTree for every request.

    Args:
        top_log_probs: [bs, L, K], normalized draft log probabilities.
        top_token_ids: [bs, L, K], matching global token IDs.
        tree_budget: number of non-root nodes to verify.
        device: destination device for runtime tensors.
        active_k: optional [bs, L] active candidate counts. Kmax remains static.
    """

    if top_log_probs.ndim != 3 or top_token_ids.shape != top_log_probs.shape:
        raise ValueError(
            "DDTree top candidate tensors must have matching [bs, L, K] shapes, "
            f"got log_probs={tuple(top_log_probs.shape)}, ids={tuple(top_token_ids.shape)}."
        )
    if tree_budget < 1:
        raise ValueError(f"DDTree tree_budget must be >= 1, got {tree_budget}.")

    bs = int(top_log_probs.shape[0])
    max_nodes = int(tree_budget) + 1

    if active_k is not None and active_k.shape != top_log_probs.shape[:2]:
        raise ValueError(
            "DDTree active_k must have shape [bs, L], got "
            f"{tuple(active_k.shape)} for candidates {tuple(top_log_probs.shape)}."
        )

    # One batched D2H copy per tensor. Performing a separate blocking copy for
    # every request makes Python heap construction disproportionately expensive.
    top_log_probs_cpu = top_log_probs.to(device="cpu", dtype=torch.float64)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    active_k_cpu = (
        None if active_k is None else active_k.to(device="cpu", dtype=torch.long)
    )

    all_tokens: List[List[int]] = []
    all_depths: List[List[int]] = []
    all_parents: List[List[int]] = []
    all_visibility: List[torch.Tensor] = []
    all_next_token: List[torch.Tensor] = []
    all_next_sibling: List[torch.Tensor] = []
    all_child_maps: List[Dict[int, Dict[int, int]]] = []

    for batch_idx in range(bs):
        tokens, depths, parents, child_map = _build_single_tree(
            top_log_probs_cpu[batch_idx],
            top_token_ids_cpu[batch_idx],
            int(tree_budget),
            None if active_k_cpu is None else active_k_cpu[batch_idx],
        )
        visibility, next_token, next_sibling = _compile_topology(parents, child_map)

        all_tokens.append(tokens)
        all_depths.append(depths)
        all_parents.append(parents)
        all_visibility.append(visibility)
        all_next_token.append(next_token)
        all_next_sibling.append(next_sibling)
        all_child_maps.append(child_map)

    return DDTreeBuildResult(
        node_token_ids=torch.tensor(
            all_tokens, dtype=torch.long, device=device
        ).view(bs, tree_budget),
        node_depths=torch.tensor(
            all_depths, dtype=torch.long, device=device
        ).view(bs, tree_budget),
        parents=torch.tensor(
            all_parents, dtype=torch.long, device=device
        ).view(bs, max_nodes),
        visibility=torch.stack(all_visibility, dim=0).to(device=device),
        retrieve_next_token=torch.stack(all_next_token, dim=0).to(device=device),
        retrieve_next_sibling=torch.stack(all_next_sibling, dim=0).to(device=device),
        child_maps=all_child_maps,
    )


def compile_ddtree_inputs(
    *,
    root_token_ids: torch.Tensor,
    start_positions: torch.Tensor,
    tree: DDTreeBuildResult,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compile flattened target verify token IDs and RoPE positions."""

    bs = int(root_token_ids.shape[0])
    if int(tree.node_token_ids.shape[0]) != bs:
        raise ValueError(
            f"DDTree batch mismatch: roots={bs}, tree={tree.node_token_ids.shape[0]}."
        )

    verify_ids = torch.cat(
        [root_token_ids.to(torch.long).unsqueeze(1), tree.node_token_ids], dim=1
    )
    positions = torch.cat(
        [
            start_positions.to(torch.long).unsqueeze(1),
            start_positions.to(torch.long).unsqueeze(1) + tree.node_depths,
        ],
        dim=1,
    )
    return verify_ids, positions


def follow_verified_tree(
    *,
    child_map: Dict[int, Dict[int, int]],
    target_predictions: Sequence[int],
) -> Tuple[List[int], int]:
    """Follow target greedy predictions through a compiled tree.

    Returns the accepted verify-node indices (including root node 0) and the
    target bonus token predicted at the final accepted node.
    """

    accepted = [0]
    current = 0
    bonus = int(target_predictions[current])

    while True:
        child = child_map.get(current, {}).get(bonus)
        if child is None:
            break
        current = int(child)
        accepted.append(current)
        bonus = int(target_predictions[current])

    return accepted, bonus
