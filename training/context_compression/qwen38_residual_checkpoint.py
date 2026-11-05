"""Model-only repeat -> residual migration; new stage gets fresh optimizer/RNG."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

from context_compression import train_qwen38_mmfine_joint as previous
from context_compression.qwen38_residual_modeling import SummaryResidualCompressor


def merge_repeat_state(repeat, fresh):
    new = {k for k in fresh if SummaryResidualCompressor.is_new_parameter(k)}
    expected = set(fresh) - new
    if set(repeat) != expected:
        raise ValueError(f"repeat keys differ: missing={expected-set(repeat)}, extra={set(repeat)-expected}")
    # Materialize new keys explicitly: FSDP SHARDED pre-load hooks require them
    # even when load_state_dict(strict=False) is requested.
    return {**repeat, **{k: fresh[k] for k in new}}


def initialize_from_repeat(source, *, compressor, rank, source_trainer, source_hash):
    payload = torch.load(source / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False)
    expected = dict(rank=rank, world_size=dist.get_world_size(),
                    schema=source_trainer["schema"], format_version=source_trainer["format_version"],
                    global_step=source_trainer["global_step"], run_contract_sha256=source_hash)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("repeat shard/trainer identity mismatch")
    repeat = payload.pop("model")
    del payload  # Never carry repeat Adam moments, data cursor, or RNG into this stage.
    state_config, optim_config = previous._checkpoint_configs()
    with FSDP.state_dict_type(compressor, StateDictType.SHARDED_STATE_DICT, state_config, optim_config):
        fresh = compressor.state_dict()
        merged = merge_repeat_state(repeat, fresh)
        del fresh, repeat
        compressor.load_state_dict(merged, strict=True)
    del merged
    previous._checkpoint_cache_barrier()
