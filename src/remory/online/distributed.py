"""Synchronous data parallelism for the residual objective's local VJP replay."""
import torch
import torch.distributed as dist


def reduce_gradients(parameters, tokens, *, bucket_bytes=64 << 20):
    """Sum token-sum gradients, including idle ranks, then normalize once."""
    parameters = list(parameters)
    if tokens <= 0:
        raise ValueError('A training step needs scored target tokens')
    presence = torch.tensor([p.grad is not None for p in parameters],
                            device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)
    views, size = [], 0

    def flush():
        nonlocal views, size
        if not views:
            return
        flat = torch.cat(views)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(tokens)
        offset = 0
        for view in views:
            view.copy_(flat[offset:offset + view.numel()])
            offset += view.numel()
        views, size = [], 0

    for p, active in zip(parameters, presence.tolist()):
        if not active:
            continue  # AdamW must not decay a globally unused parameter.
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        for view in p.grad.reshape(-1).split(max(1, bucket_bytes // p.element_size())):
            width = view.numel() * view.element_size()
            if views and (size + width > bucket_bytes or views[0].dtype != view.dtype):
                flush()
            views.append(view)
            size += width
    flush()
