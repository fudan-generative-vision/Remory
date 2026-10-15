"""Audited full-state checkpoint and FP32 gradient reduction helpers for Stage I."""
import dataclasses
import hashlib
import json
import os
import random
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig, StateDictType
from context_compression import train_qwen38_mmfine_joint as old
SCHEMA = "qwen38_pretrained_dflash2_stage1_3b_four_gpu_v1"

def parameter_fingerprint(model):
    """Cheap rank consistency check before FSDP singleton-group wrapping."""
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(f"{name}:{tuple(value.shape)}".encode())
        flat = value.detach().reshape(-1)
        h.update(torch.cat((flat[:64], flat[-64:])).float().cpu().numpy().tobytes())
    return h.hexdigest()


def sum_replica_gradients(parameters, *, bucket_bytes=64 * 1024 * 1024):
    """Sum full FP32 replica gradients; do not decay globally unused parameters."""
    parameters = list(parameters)
    presence = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)
    active = []
    for p, exists in zip(parameters, presence.tolist()):
        if exists:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            active.append(p.grad)
    # Split individual large parameters too; never allocate a full model-sized bucket.
    views, size = [], 0
    def flush():
        nonlocal views, size
        if not views:
            return
        flat = torch.cat(views)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for view in views:
            view.copy_(flat[offset:offset + view.numel()])
            offset += view.numel()
        views, size = [], 0
    for grad in active:
        for view in grad.reshape(-1).split(max(1, bucket_bytes // grad.element_size())):
            if size + view.numel() * view.element_size() > bucket_bytes:
                flush()
            views.append(view)
            size += view.numel() * view.element_size()
    flush()


def singleton_group(rank, world):
    result = None
    for i in range(world):
        group = dist.new_group([i])
        if rank == i:
            result = group
    return result


def full_state_context(model):
    rank0_only = dist.get_world_size(model.process_group) > 1
    return FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
                               FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only))


def load_full_optimizer(model, optimizer, path, *, fused):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    for group in state["param_groups"]:
        group.update(fused=fused, foreach=False)
    with full_state_context(model):
        converted = FSDP.optim_state_dict_to_load(model, optimizer, state, group=model.process_group)
    optimizer.load_state_dict(converted)
    del state, converted


def restore_rng(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["rng_cpu"])
    torch.cuda.set_rng_state(state["rng_cuda"])
    random.setstate(state["rng_python"])


def save_full_checkpoint(output, model, optimizer, *, mode, rank, step, cursor, contract_hash):
    destination = output / f"checkpoint-{step:08d}"
    staging = output / f".{destination.name}.staging"
    if rank == 0:
        if destination.exists() or staging.exists():
            raise FileExistsError(destination)
        staging.mkdir()
    dist.barrier()
    torch.save({"rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                "rng_python": random.getstate()}, staging / f"rng-{rank}.pt")
    if mode == "tp" or rank == 0:
        with full_state_context(model):
            weights = model.state_dict()
            moments = FSDP.optim_state_dict(model, optimizer, group=model.process_group)
        if rank == 0:
            torch.save(weights, staging / "model.pt")
            torch.save(moments, staging / "optimizer.pt")
        del weights, moments
    dist.barrier()
    if rank == 0:
        old._atomic_json(staging / "run_config.json", json.loads((output / "run_config.json").read_text()))
        old._atomic_json(staging / "trainer_state.json", dict(schema=SCHEMA, global_step=step,
            cursor=dataclasses.asdict(cursor), run_contract_sha256=contract_hash, world_size=dist.get_world_size()))
        os.rename(staging, destination)
    dist.barrier()
    return destination


