"""CPU-only model export for repeat checkpoint migration to another FSDP world size."""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
from context_compression import train_qwen38_mmfine_joint as old
from context_compression.export_qwen36_fsdp_checkpoint import _local_model_part, _assemble
from context_compression.qwen38_allturn_data import digest
from context_compression.grepseek_modeling import GrepSeekDFlashCompressor

SCHEMA = "qwen38_repeat_model_only_full_state_v1"
REPEAT_SCHEMAS = {old.CHECKPOINT_SCHEMA, "qwen38_mmfine_full_repeat_reverse_kl_frozen_reader_v1", "qwen38_long_document_full_repeat_reverse_kl_frozen_reader_v1"}

def validate_state(state, expected):
    if set(state) != set(expected):raise ValueError("repeat model key mismatch")
    for name, value in state.items():
        if not isinstance(value,torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"repeat model tensor mismatch: {name}")
        if not all(bool(torch.isfinite(x).all()) for x in value.reshape(-1).split(4*1024*1024)):
            raise ValueError(f"nonfinite repeat tensor: {name}")

def read_export(directory, *, source, source_trainer, source_hash, verify_weights=True):
    directory=Path(directory).resolve();r=json.loads((directory/"manifest.json").read_text())
    expected=dict(schema=SCHEMA,source_checkpoint=str(Path(source).resolve()),source_run_contract_sha256=source_hash,source_step=source_trainer["global_step"])
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError("repeat export source identity mismatch")
    if r.get("model_file")!="compressor.pt":raise ValueError("unexpected exported model filename")
    if verify_weights and digest(directory/r["model_file"])!=r["model_sha256"]:raise ValueError("repeat export weight checksum mismatch")
    return r


def initialize_base_from_export(base, directory, receipt):
    """Load only the repeat backbone; preserve seeded new residual parameters."""
    from context_compression.qwen38_residual_checkpoint import merge_repeat_state
    weights = torch.load(Path(directory) / receipt["model_file"],
                         map_location="cpu", weights_only=True, mmap=True)
    base.load_state_dict(merge_repeat_state(weights, base.state_dict()), strict=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);dist.init_process_group("gloo");rank=dist.get_rank();world=dist.get_world_size()
    try:
        source=a.checkpoint.resolve();output=a.output_dir.resolve();staging=output.with_name("."+output.name+".staging")
        trainer=json.loads((source/"trainer_state.json").read_text());config=old._source_run_contract(source);source_hash=old._canonical_json_sha256(config)
        if trainer["schema"] not in REPEAT_SCHEMAS or source_hash!=trainer["run_contract_sha256"] or len(trainer["rank_files"])!=world:raise ValueError("invalid repeat source/world size")
        if output.exists() or staging.exists():raise FileExistsError(output)
        if rank==0:staging.mkdir(parents=True)
        dist.barrier();rank_path=source/f"rank-{rank:05d}.pt";before=digest(rank_path)
        payload=torch.load(rank_path,map_location="cpu",weights_only=False,mmap=True)
        expected=dict(rank=rank,world_size=world,schema=trainer["schema"],format_version=trainer["format_version"],global_step=trainer["global_step"],run_contract_sha256=source_hash)
        if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError("repeat shard identity mismatch")
        part=_local_model_part(payload["model"]);torch.save(part,staging/f"part-{rank}.pt");del part,payload
        if digest(rank_path)!=before:raise ValueError("repeat shard changed during export")
        receipts=[None]*world;dist.all_gather_object(receipts,dict(file=rank_path.name,sha256=before))
        if rank==0:
            parts=[torch.load(staging/f"part-{i}.pt",map_location="cpu",weights_only=True,mmap=True) for i in range(world)]
            full=_assemble(parts)
            with torch.device("meta"):
                expected=GrepSeekDFlashCompressor.canonical_qwen36_27b(output_embedding_rms=config["compressor"]["output_embedding_rms"],gradient_checkpointing=False).float().state_dict()
            validate_state(full,expected);del parts,expected
            torch.save(full,staging/"compressor.pt");numel=sum(t.numel() for t in full.values());del full
            receipt=dict(schema=SCHEMA,source_checkpoint=str(source),source_run_contract_sha256=source_hash,source_step=trainer["global_step"],source_world_size=world,source_shards=receipts,source_metadata_sha256={n:digest(source/n) for n in ["trainer_state.json","run_config.json"]},compressor=config["compressor"],model_file="compressor.pt",model_sha256=digest(staging/"compressor.pt"),model_numel=numel,optimizer_exported=False)
            (staging/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
            for i in range(world):(staging/f"part-{i}.pt").unlink()
            if output.exists():raise FileExistsError(output)
            os.rename(staging,output);print(json.dumps(receipt),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()

if __name__=="__main__":main()
