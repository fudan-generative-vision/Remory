"""Long-document full repeat: frozen TP2 Qwen, randomly initialized FSDP Compressor."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.tensorboard import SummaryWriter

from context_compression import train_qwen38_mmfine_joint as old
from context_compression.grepseek_modeling import GrepSeekDFlashCompressor
from context_compression.qwen38_long_repeat_data import load_dataset, training_order, load_example
from context_compression.qwen38_multimodal_target import (
    FrozenQwen38MultimodalTarget, load_qwen38_multimodal_model, load_qwen38_processor,
)
from context_compression.qwen38_long_repeat_kl import full_repeat_kl, render_full_repeat

SCHEMA = "qwen38_long_document_full_repeat_reverse_kl_frozen_reader_v1"
FORMAT_VERSION = 1


def implementation_receipts():
    names = ["train_qwen38_long_repeat.py", "qwen38_long_repeat_kl.py", "qwen38_long_repeat_data.py",
             "qwen38_reverse_kl.py", "train_qwen38_mmfine_joint.py",
             "mmfine_reasoning_data.py", "qwen38_multimodal_target.py",
             "grepseek_modeling.py", "qwen38_allturn_training.py"]
    return {n: old._sha256(Path(__file__).with_name(n)) for n in names}


def save_checkpoint(output, compressor, optimizer, residency, step, cursor, contract_hash, rank):
    destination = output / f"checkpoint-{step:08d}"
    staging = output / f".{destination.name}.staging"
    if rank == 0:
        if destination.exists() or staging.exists():
            raise FileExistsError(destination)
        staging.mkdir()
    old._checkpoint_cache_barrier()
    try:
        residency.prepare_forward(optimizer)
        config, optim_config = old._checkpoint_configs()
        with FSDP.state_dict_type(compressor, StateDictType.SHARDED_STATE_DICT, config, optim_config):
            model = compressor.state_dict()
            old._checkpoint_cache_barrier()
            optim = FSDP.optim_state_dict(compressor, optimizer)
    finally:
        residency.offload(optimizer, reason="after_save", tolerate_partial=True)
        old._checkpoint_cache_barrier()
    payload = dict(format_version=FORMAT_VERSION, schema=SCHEMA, rank=rank,
                   world_size=dist.get_world_size(), global_step=step,
                   run_contract_sha256=contract_hash, model=model, optimizer=optim,
                   rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
                   rng_python=random.getstate())
    torch.save(payload, staging / f"rank-{rank:05d}.pt")
    del payload, model, optim
    dist.barrier()
    if rank == 0:
        old._atomic_json(staging / "run_config.json", json.loads((output / "run_config.json").read_text()))
        old._atomic_json(staging / "trainer_state.json", dict(
            format_version=FORMAT_VERSION, schema=SCHEMA, global_step=step,
            cursor=dataclasses.asdict(cursor), run_contract_sha256=contract_hash,
            rank_files=[f"rank-{i:05d}.pt" for i in range(dist.get_world_size())]))
        os.replace(staging, destination)
    dist.barrier()
    return destination


class GPUMomentResidency:
    """Keep sharded Adam states on their parameter devices across steps."""
    def restore(self, optimizer):
        for p, state in old.LowPeakAdamMomentResidency._items(optimizer):
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = state[key].to(p.device)

    def prepare_forward(self, optimizer):
        self.restore(optimizer)

    def offload(self, optimizer, **kwargs):
        # The checkpoint machinery requests this hook after saving/loading.
        self.restore(optimizer)

    def receipt(self):
        return {"policy": "gpu_resident_sharded_adam_v1"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("target-model", "manifest", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--random-init", action="store_true")
    source.add_argument("--resume-from", type=Path)
    p.add_argument("--embedding-rms", type=float, required=True,
                   help="RMS measured from the frozen target token embeddings")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-accumulation", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--maximum-source-tokens", type=int, default=65536)
    p.add_argument("--projection-tokens", type=int, default=64)
    p.add_argument("--offload-after-tokens", type=int, default=131072)
    p.add_argument("--save-interval", type=int, default=25)
    p.add_argument("--keep-last-checkpoints", type=int, default=3)
    p.add_argument("--max-additional-steps", type=int, default=0)
    p.add_argument("--canary-lengths", default="",
                   help="Real-source forward/backward checks, no optimizer updates")
    return p


def main():
    args = parser().parse_args()
    if min(args.epochs, args.grad_accumulation, args.projection_tokens, args.offload_after_tokens) <= 0:
        raise ValueError("invalid budget")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(8)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    if dist.get_world_size() != 2:
        raise ValueError("requires TP2/FSDP2")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    output = args.output_dir.resolve()
    manifest, db = load_dataset(args.manifest, args.target_model)
    if manifest["selection"]["maximum_tokens"] > args.maximum_source_tokens:
        raise ValueError("dataset exceeds configured source budget")
    tokenizer_receipt = old._tokenizer_files_receipt(args.target_model)
    previous_contract = old._source_run_contract(args.resume_from) if args.resume_from else None
    base = GrepSeekDFlashCompressor.canonical_qwen36_27b(
        output_embedding_rms=args.embedding_rms, compression_ratio=16,
        local_window=2048, gradient_checkpointing=True).float()
    compressor_config = base.export_config()
    initialization = dict(kind="random", seed=args.seed, weights="torch_normal_init",
                          optimizer="fresh", global_step=0, data_cursor="reset",
                          output_embedding_rms_source="frozen_target_token_embeddings")
    contract = dict(
        schema=SCHEMA, format_version=FORMAT_VERSION,
        target_model=str(args.target_model.resolve()), target_lora=False,
        target_trainable_parameters=0, manifest_sha256=manifest["manifest_sha256"],
        dataset=manifest, tokenizer=tokenizer_receipt, compressor=compressor_config,
        compressor_initialization=initialization,
        objective="full_vocabulary_reverse_KL_student_to_raw_teacher_T1_no_CE",
        repeat=dict(fields=["long_document_window"],
                    source_chunk_tokens=args.maximum_source_tokens, source_block_tokens=1024, compression_ratio=16,
                    target_window_tokens=None, target_context_tokens=0,
                    target="ALL_source_chunk_tokens_plus_im_end", empty_thinking=True,
                    teacher="raw_source+prompt->response", student="soft_tokens+prompt->response",
                    coverage="each_selected_document_window_each_token_once_per_epoch",
                    order="length_buckets_interleaved_without_oversampling_seed_plus_epoch"),
        position_weighting=dict(alpha=1.0, tau=64.0, power=0.5,
                                formula="(1+(1+p/64)^(-0.5))/full_response_mean",
                                coordinate="original_source_offset_without_chunk_reset"),
        optimizer=dict(type="AdamW", learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                       betas=[0.9, 0.95], eps=1e-8, weight_decay=0.01,
                       grad_accumulation=args.grad_accumulation, normalization="scored_token_mean",
                       moment_residency="GPU_resident_sharded"),
        epochs=args.epochs, seed=args.seed, projection_tokens=args.projection_tokens,
        offload_after_tokens=args.offload_after_tokens, implementation=implementation_receipts())
    contract_hash = old._canonical_json_sha256(contract)
    if previous_contract and contract != previous_contract:
        raise ValueError("exact full-repeat resume contract changed")
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume_from:
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        old._atomic_json(output / "run_config.json", {**contract,
            "run_contract_sha256": contract_hash, "output_dir": str(output),
            "tensorboard_dir": str(output / "tensorboard"),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "init_compressor": None})
        print(json.dumps(dict(phase="loading_frozen_model", source=initialization,
                              training_documents=manifest["counts"]["train"])), flush=True)
    processor = load_qwen38_processor(args.target_model)
    model = load_qwen38_multimodal_model(str(args.target_model), dtype=torch.bfloat16,
                                        attn_implementation="sdpa", tp_plan="auto")
    target = FrozenQwen38MultimodalTarget(model, gradient_checkpointing=True)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("target is not frozen")
    compressor = FSDP(base, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      buffer_dtype=torch.bfloat16, cast_forward_inputs=True),
        use_orig_params=True, sync_module_states=True, limit_all_gathers=True)
    compressor.train()
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, foreach=False)
    residency = GPUMomentResidency()
    step, cursor = 0, old.QueueCursor(0, 0, 0, 0)
    if args.resume_from:
        step, cursor = old._load_checkpoint(args.resume_from.resolve(), compressor=compressor,
            optimizer=optimizer, residency=residency, rank=rank,
            checkpoint_format_version=FORMAT_VERSION, checkpoint_schema=SCHEMA,
            checkpoint_run_contract_sha256=contract_hash)
    optimizer.zero_grad(set_to_none=True)
    initial_step = step

    for width in (int(v) for v in args.canary_lengths.split(",") if v):
        row = db.execute("SELECT id FROM examples WHERE split='train' AND token_count>=? ORDER BY token_count,id LIMIT 1", (width,)).fetchone()
        if row is None or width > args.maximum_source_tokens:
            raise ValueError("no real source covers the requested canary length")
        example = load_example(db, row[0])
        rendered = render_full_repeat(processor, example["token_ids"][:width])
        torch.cuda.reset_peak_memory_stats()
        probe_started = time.monotonic()
        if rank == 0:
            print(json.dumps(dict(phase="real_canary_start", source_tokens=width,
                                  document_sha256=example["document_sha256"])), flush=True)
        loss, metrics = full_repeat_kl(rendered, target=target, compressor=compressor,
            source_offset=example["token_start"], projection_tokens=args.projection_tokens,
            offload_after_tokens=args.offload_after_tokens,
            maximum_source_tokens=args.maximum_source_tokens)
        loss.backward()
        norm = float(compressor.clip_grad_norm_(1.0))
        if not math.isfinite(norm) or norm <= 0 or not math.isfinite(metrics["reverse_kl"]):
            raise RuntimeError("real-source backward did not produce a finite nonzero gradient")
        report = dict(passed=True, optimizer_updated=False, **metrics, grad_norm=norm,
                      document_sha256=example["document_sha256"],
                      elapsed_seconds=time.monotonic()-probe_started,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        del loss, rendered
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if rank == 0:
            old._atomic_json(output / f"real_canary_{width}.json", report)
            print(json.dumps(dict(phase="real_canary_passed", **report)), flush=True)

    writer = SummaryWriter(str(output / "tensorboard"), purge_step=step+1 if args.resume_from else None) if rank == 0 else None
    pending = dict(tokens=0, responses=0, loss_sum=0.0, agreement_sum=0.0,
                   max_source_tokens=0, max_response_tokens=0, max_soft_tokens=0)
    started = time.monotonic()
    stopping, stop, last_saved = False, False, -1
    checkpoint = None

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGUSR1, request_stop)

    def save():
        nonlocal checkpoint, last_saved
        checkpoint = save_checkpoint(output, compressor, optimizer, residency, step, cursor, contract_hash, rank)
        last_saved = step
        old._prune_checkpoints(output, keep_last=args.keep_last_checkpoints, rank=rank)

    def update():
        nonlocal pending, step
        if not pending["tokens"]:
            return
        old._normalize_gradients(compressor.parameters(), pending["tokens"])
        norm = float(compressor.clip_grad_norm_(1.0))
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError("invalid compressor gradient")
        lr = old._learning_rate(step, base=args.learning_rate, warmup=args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        residency.restore(optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if not old._parameters_finite(compressor):
            raise FloatingPointError("nonfinite parameters")
        row = dict(global_step=step, loss=pending["loss_sum"]/pending["tokens"],
            reverse_kl=pending["loss_sum"]/pending["tokens"],
            top1_agreement=pending["agreement_sum"]/pending["tokens"],
            grad_norm=norm, learning_rate=lr, elapsed_seconds=time.monotonic()-started,
            cursor=dataclasses.asdict(cursor), target_lora=False, target_trainable_parameters=0,
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            moment_residency=residency.receipt(),
            **{k:v for k,v in pending.items() if not k.endswith("_sum")})
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            old._write_tensorboard_summary(writer, row, global_step=step)
            print(json.dumps(row, allow_nan=False), flush=True)
        if args.save_interval and step % args.save_interval == 0:
            save()
        pending = {k:0 for k in pending}

    start_cursor = cursor
    for epoch in range(start_cursor.epoch, args.epochs):
        order = training_order(db, args.seed, epoch)
        first = start_cursor.event_index if epoch == start_cursor.epoch else 0
        if first >= len(order) or start_cursor.shard_index or start_cursor.chunk_index:
            raise ValueError("resume cursor outside immutable document order")
        for index in range(first, len(order)):
            example = load_example(db, order[index])
            identities = [None] * 2
            identity = (epoch, index, example["id"], old._canonical_json_sha256(example["token_ids"]))
            dist.all_gather_object(identities, identity)
            if any(x != identity for x in identities):
                raise RuntimeError("TP ranks read different source tokens")
            residency.prepare_forward(optimizer)
            rendered = render_full_repeat(processor, example["token_ids"])
            if rank == 0:
                print(json.dumps(dict(phase="response_start", epoch=epoch, index=index,
                                      source_tokens=len(example["token_ids"]), id=example["id"])), flush=True)
            response_started = time.monotonic()
            loss, metrics = full_repeat_kl(rendered, target=target, compressor=compressor,
                source_offset=example["token_start"], projection_tokens=args.projection_tokens,
                offload_after_tokens=args.offload_after_tokens,
                maximum_source_tokens=args.maximum_source_tokens)
            count = metrics["tokens"]
            (loss * count).backward()
            del loss, rendered
            if rank == 0:
                print(json.dumps(dict(phase="response_backward_complete", index=index,
                                      elapsed_seconds=time.monotonic()-response_started, **metrics)), flush=True)
            pending["tokens"] += count
            pending["responses"] += 1
            pending["loss_sum"] += metrics["reverse_kl"] * count
            pending["agreement_sum"] += metrics["top1_agreement"] * count
            for key, value in [("max_source_tokens", metrics["source_tokens"]),
                               ("max_response_tokens", count), ("max_soft_tokens", metrics["soft_tokens"])]:
                pending[key] = max(pending[key], value)
            cursor = old.QueueCursor(epoch, 0, index + 1, 0) if index + 1 < len(order) else old.QueueCursor(epoch + 1, 0, 0, 0)
            stop_flag = torch.tensor(int(stopping or (output / "STOP_AFTER_RESPONSE").exists()), device=device)
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if pending["responses"] >= args.grad_accumulation or bool(stop_flag) or index + 1 == len(order):
                update()
                stop = bool(stop_flag) or (args.max_additional_steps > 0 and step-initial_step >= args.max_additional_steps)
            if stop:
                break
        if stop:
            break
    db.close()
    if step != last_saved:
        save()
    if rank == 0:
        old._atomic_json(output/"completion.json", dict(global_step=step,
            checkpoint=str(checkpoint), cursor=dataclasses.asdict(cursor),
            all_epochs_complete=cursor.epoch >= args.epochs,
            stopped_on_request=stopping or (output/"STOP_AFTER_RESPONSE").exists()))
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
