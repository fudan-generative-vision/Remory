"""Inference-only packed GDN tiling with independent recurrent state per request.

This bounds FLA's [chunks, heads, V, K] intermediate, not the source length.
The caller still supplies every source row and captures every output row.
Tile boundaries align with the original 64-token FLA chunks. Prefix-cache
checkpoints are copied from the same h rows before each temporary is released.
"""
import torch

CAPABILITY = "qwen38_residual_bounded_gdn_prefill_v1"
ALL_PREFILL_CAPABILITY = "qwen38_bounded_gdn_all_prefill_v2"
TILE_TOKENS = 2048


def bounded_gdn_prefill(kernel, *, q, k, v, g, beta, ssm_states, cache_indices,
                        query_start_loc, track=None, tile_tokens=TILE_TOKENS):
    if type(tile_tokens) is not int or tile_tokens <= 0 or tile_tokens % 64:
        raise ValueError("GDN tile must be a positive multiple of the physical 64-token chunk")
    if q.ndim != 4 or q.shape[0] != 1 or query_start_loc.ndim != 1 or cache_indices.ndim != 1:
        raise ValueError("bounded GDN requires flattened packed sequences")
    boundaries = query_start_loc.tolist()
    indices = cache_indices.tolist()
    if (len(boundaries) != len(indices) + 1 or not boundaries or boundaries[0] != 0
            or any(type(n) is not int for n in boundaries + indices)
            or any(a > b for a, b in zip(boundaries, boundaries[1:]))
            or boundaries[-1] > q.shape[1]):
        raise ValueError("invalid packed GDN sequence boundaries")
    active = [idx for idx, a, b in zip(indices, boundaries, boundaries[1:]) if a != b]
    if (len(set(active)) != len(active)
            or any(idx < 0 or idx >= ssm_states.shape[0] for idx in active)):
        raise ValueError("packed GDN requests require distinct valid recurrent state slots")
    if any(t.shape[:2] != q.shape[:2] for t in (k, v, g, beta)):
        raise ValueError("packed GDN input row counts disagree")
    # Graph/DP padding, if present, is not a request and must not touch a state.
    output = torch.empty_like(v)
    output[:, boundaries[-1]:].zero_()
    tracked = []
    if track is not None and track.has_mamba_track_mask:
        sources, destinations = track.track_ssm_h_src.tolist(), track.track_ssm_h_dst.tolist()
        total_chunks = sum((b - a + 63) // 64 for a, b in zip(boundaries, boundaries[1:]))
        if (len(sources) != len(destinations) or any(s < 0 or s >= total_chunks for s in sources)
                or any(d < 0 or d >= ssm_states.shape[0] for d in destinations)):
            raise ValueError("invalid packed GDN prefix checkpoint")
        final_src, final_dst = track.track_ssm_final_src.tolist(), track.track_ssm_final_dst.tolist()
        if (len(final_src) != len(final_dst)
                or any(i < 0 or i >= ssm_states.shape[0] for i in final_src + final_dst)):
            raise ValueError("invalid packed GDN final checkpoint")
        tracked = list(zip(sources, destinations))
    checkpoints = [None] * len(tracked)
    chunk_offset = 0
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        # Tile boundaries are relative to EACH request, not the packed tensor:
        # a non-64-aligned previous request must not shift the next one's chunks.
        for left in range(start, end, tile_tokens):
            right = min(left + tile_tokens, end)
            cu = query_start_loc.new_tensor([0, right - left])
            # FLA updates only this request's FP32 state, carrying it unchanged
            # into the next tile. Other requests' slots are never used as seeds.
            part, _, h = kernel(q=q[:, left:right], k=k[:, left:right], v=v[:, left:right],
                g=g[:, left:right], beta=beta[:, left:right], initial_state=ssm_states,
                initial_state_indices=cache_indices[seq:seq + 1], cu_seqlens=cu,
                head_first=False, use_qk_l2norm_in_kernel=True)
            output[:, left:right].copy_(part)
            for i, (source, _) in enumerate(tracked):
                local = source - chunk_offset - (left - start) // 64
                if 0 <= local < h.shape[1]:
                    checkpoints[i] = h[0, local].clone()
            del part, h
        chunk_offset += (end - start + 63) // 64
    # Match the original backend's post-forward tracking order, even if a
    # destination aliases an active slot. Never perturb the carried state.
    for (_, destination), checkpoint in zip(tracked, checkpoints):
        if checkpoint is None:
            raise RuntimeError("packed GDN prefix checkpoint was not produced")
        ssm_states[destination].copy_(checkpoint)
    if track is not None and track.has_mamba_track_mask and track.track_ssm_final_src.numel():
        ssm_states[track.track_ssm_final_dst] = ssm_states[track.track_ssm_final_src]
    return output, None, None
