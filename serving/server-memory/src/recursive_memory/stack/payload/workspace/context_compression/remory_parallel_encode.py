"""Distribute independent residual nodes without changing their arithmetic.

The target still sees the complete causal prefix. Each node retains its own
1024-to-64 compressor forward, including the final slot mixer. Only independent
nodes are distributed; every TP worker receives the ordered complete result.
"""
import torch
import torch.distributed as dist


def encode_blocks(compressor, aux, *, start, end, block_tokens, compression_ratio,
                  hidden_size, summary, summary_mask, depths, group=None,
                  parallel=True):
    count = (end - start + block_tokens - 1) // block_tokens
    if count < 1 or len(depths) != count:
        raise ValueError('one input depth is required for each nonempty source block')
    parameter = next(compressor.parameters())
    device = parameter.device
    slots = block_tokens // compression_ratio
    distributed = parallel and count > 1 and dist.is_initialized()
    world = dist.get_world_size(group) if distributed else 1
    rank = dist.get_rank(group) if distributed else 0
    local_count = (count + world - 1) // world
    packed = torch.zeros((local_count, slots, hidden_size), device=device, dtype=parameter.dtype)
    failure = None
    try:
        with torch.inference_mode():
            for local_index, block_index in enumerate(range(rank, count, world)):
                left = start + block_index * block_tokens
                right = min(end, left + block_tokens)
                # Preserve the serial kernel shapes and bounded feature staging.
                features = torch.cat([h[left:right] for h in aux], dim=-1).to(device=device)[None]
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
                encoded = compressor(
                    features, mask,
                    source_token_lengths=torch.tensor([right-left], device=device),
                    slot_token_lengths=torch.tensor([block_tokens], device=device),
                    summary_features=summary, summary_attention_mask=summary_mask,
                    input_depth=depths[block_index])
                memory = encoded.embeddings[encoded.attention_mask]
                if memory.shape != (slots, hidden_size):
                    raise ValueError('residual node violated fixed 1024-to-64 geometry')
                packed[local_index].copy_(memory)
    except Exception as exc:
        if world == 1:
            raise
        failure = type(exc).__name__ + ': ' + str(exc)
    if world > 1:
        # A failed rank must not leave peers waiting in the tensor collective.
        errors = [None] * world
        dist.all_gather_object(errors, failure, group=group)
        if any(error is not None for error in errors):
            raise RuntimeError('parallel residual encode failed: ' + repr(errors))
        pieces = [torch.empty_like(packed) for _ in range(world)]
        dist.all_gather(pieces, packed, group=group)
        packed = torch.stack(pieces, dim=1).flatten(0, 1)[:count]
    return packed.reshape(count * slots, hidden_size)
