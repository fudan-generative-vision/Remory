"""Consolidate FSDP residual model AND Adam state for an audited execution migration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from context_compression import train_qwen38_mmfine_joint as old
from context_compression.export_qwen36_fsdp_checkpoint import _assemble, _local_model_part
from context_compression.qwen38_allturn_data import digest

SCHEMA = "qwen38_residual_portable_training_state_v1"


def flatten_optimizer(state):
    return {json.dumps([name, field]): value
            for name, fields in state["state"].items() for field, value in fields.items()}


def unflatten_optimizer(values, groups):
    state = {}
    for key, value in values.items():
        name, field = json.loads(key)
        state.setdefault(name, {})[field] = value
    return {"state": state, "param_groups": groups}


def read_export(path, source):
    path, source = Path(path).resolve(), Path(source).resolve()
    receipt = json.loads((path / "manifest.json").read_text())
    trainer = json.loads((source / "trainer_state.json").read_text())
    if (receipt["schema"] != SCHEMA or receipt["source_checkpoint"] != str(source)
            or receipt["global_step"] != trainer["global_step"]
            or receipt["run_contract_sha256"] != trainer["run_contract_sha256"]):
        raise ValueError("portable residual state source mismatch")
    for name, checksum in receipt["files"].items():
        if digest(path / name) != checksum:
            raise ValueError(f"portable residual checksum mismatch: {name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    source, output = args.checkpoint.resolve(), args.output_dir.resolve()
    staging = output.with_name("." + output.name + ".staging")
    trainer = json.loads((source / "trainer_state.json").read_text())
    config = old._source_run_contract(source)
    config_hash = old._canonical_json_sha256(config)
    if (trainer["schema"] != "qwen38_compact_residual_context_sorted_response_rkl_tp_fsdp_v4"
            or config_hash != trainer["run_contract_sha256"] or len(trainer["rank_files"]) != world):
        raise ValueError("unsupported residual source or world size")
    if rank == 0:
        if output.exists() or staging.exists():
            raise FileExistsError(output)
        staging.mkdir(parents=True)
    dist.barrier()
    source_file = source / trainer["rank_files"][rank]
    checksum = digest(source_file)
    payload = torch.load(source_file, map_location="cpu", weights_only=False, mmap=True)
    identity = dict(rank=rank, world_size=world, schema=trainer["schema"],
                    global_step=trainer["global_step"], run_contract_sha256=config_hash)
    if any(payload.get(k) != v for k, v in identity.items()):
        raise ValueError("residual rank identity mismatch")
    torch.save({"model": _local_model_part(payload["model"]),
                "optimizer": _local_model_part(flatten_optimizer(payload["optimizer"])),
                "param_groups": payload["optimizer"]["param_groups"]}, staging / f"part-{rank}.pt")
    torch.save({k: payload[k] for k in ("rng_cpu", "rng_cuda", "rng_python")},
               staging / f"rng-{rank}.pt")
    del payload
    if digest(source_file) != checksum:
        raise ValueError("source checkpoint changed during consolidation")
    receipts = [None] * world
    dist.all_gather_object(receipts, {"file": source_file.name, "sha256": checksum})
    if rank == 0:
        parts = [torch.load(staging / f"part-{i}.pt", map_location="cpu", weights_only=True, mmap=True)
                 for i in range(world)]
        full_model = _assemble([p["model"] for p in parts])
        torch.save(full_model, staging / "model.pt")
        shapes = {k: v.shape for k, v in full_model.items()}
        del full_model
        groups = parts[0]["param_groups"]
        if any(p["param_groups"] != groups for p in parts[1:]):
            raise ValueError("optimizer parameter groups differ across ranks")
        optimizer = unflatten_optimizer(_assemble([p["optimizer"] for p in parts]), groups)
        for name, fields in optimizer["state"].items():
            for field in ("exp_avg", "exp_avg_sq"):
                if fields[field].shape != shapes[name]:
                    raise ValueError(f"optimizer shape mismatch: {name}/{field}")
                if not all(bool(torch.isfinite(t).all()) for t in fields[field].reshape(-1).split(4 * 1024 * 1024)):
                    raise ValueError(f"nonfinite optimizer state: {name}/{field}")
        torch.save(optimizer, staging / "optimizer.pt")
        del optimizer, parts
        (staging / "source_contract.json").write_text(json.dumps(config, indent=2) + "\n")
        names = ["model.pt", "optimizer.pt", "source_contract.json"] + [f"rng-{i}.pt" for i in range(world)]
        receipt = dict(schema=SCHEMA, source_checkpoint=str(source), source_world_size=world,
                       global_step=trainer["global_step"], cursor=trainer["cursor"],
                       run_contract_sha256=config_hash, source_shards=receipts,
                       optimizer_exported=True, files={n: digest(staging / n) for n in names})
        (staging / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for i in range(world):
            (staging / f"part-{i}.pt").unlink()
        os.rename(staging, output)
        print(json.dumps(receipt), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
