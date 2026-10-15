"""Compact residual: frozen TP2/TP4 reader, full-answer reverse KL, FSDP C."""

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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.event_pb2 import Event, SessionLog

from context_compression import train_qwen38_mmfine_joint as previous
from context_compression.qwen38_residual_modeling import SummaryResidualCompressor
from context_compression.qwen38_residual_data import open_summary_manifest
from context_compression.qwen38_residual_streaming import StreamingSummaryCache, validate_boundary
from context_compression.prepare_qwen38_residual_summaries import validate_dataset
from context_compression.qwen38_residual_checkpoint import initialize_from_repeat
from context_compression.export_qwen38_repeat_state import read_export, initialize_base_from_export
from context_compression.train_qwen38_long_repeat import GPUMomentResidency
from context_compression.prepare_qwen38_summary_selection import load_selection
from context_compression.qwen38_allturn_data import (
    digest,
)
from context_compression.qwen38_residual_curriculum import (
    load_context_order, load_ordered_response, validate_response_cursor,
)
from context_compression.qwen38_residual_training import (
    ResidualObjective,
)
from context_compression.qwen38_multimodal_target import (
    FrozenQwen38MultimodalTarget,
    load_qwen38_multimodal_model,
)

SCHEMA = "qwen38_compact_residual_context_sorted_response_rkl_tp_fsdp_v4"
FORMAT_VERSION = 2


def implementation_receipts() -> dict:
    root = Path(__file__).resolve().parent
    names = [
        "train_qwen38_allturn_residual_kl.py",
        "export_qwen38_repeat_state.py", "export_qwen36_fsdp_checkpoint.py",
        "prepare_qwen38_summary_selection.py",
        "train_qwen38_long_repeat.py", "qwen38_long_repeat_data.py", "qwen38_long_repeat_kl.py",
        "qwen38_residual_modeling.py", "qwen38_residual_training.py",
        "qwen38_residual_pyramid.py", "qwen38_residual_data.py",
        "qwen38_residual_checkpoint.py", "prepare_qwen38_residual_summaries.py",
        "qwen38_residual_curriculum.py",
        "qwen38_residual_streaming.py",
        "qwen38_reverse_kl.py",
        "qwen38_allturn_training.py",
        "qwen38_allturn_data.py",
        "qwen38_long_history_data.py",
        "train_qwen38_mmfine_joint.py",
        "qwen38_multimodal_target.py",
        "grepseek_modeling.py",
        "losses.py",
        "qwen38_coding_trajectory_data.py",
    ]
    return {name: digest(root / name) for name in names}


def save_checkpoint(
    output: Path,
    *,
    compressor,
    optimizer,
    residency,
    step: int,
    cursor: previous.QueueCursor,
    contract_hash: str,
    rank: int,
    summary_prefix: dict | None = None,
) -> Path:
    destination = output / f"checkpoint-{step:08d}"
    staging = output / f".{destination.name}.staging"
    if rank == 0:
        if destination.exists() or staging.exists():
            raise FileExistsError(destination)
        staging.mkdir()
    previous._checkpoint_cache_barrier()
    try:
        residency.prepare_forward(optimizer)
        state_config, optim_config = previous._checkpoint_configs()
        with FSDP.state_dict_type(
            compressor, StateDictType.SHARDED_STATE_DICT, state_config, optim_config
        ):
            model = compressor.state_dict()
            previous._checkpoint_cache_barrier()
            optim = FSDP.optim_state_dict(compressor, optimizer)
    finally:
        residency.offload(optimizer, reason="after_save", tolerate_partial=True)
        previous._checkpoint_cache_barrier()
    payload = {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "global_step": step,
        "run_contract_sha256": contract_hash,
        "model": model,
        "optimizer": optim,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "rng_python": random.getstate(),
    }
    torch.save(payload, staging / f"rank-{rank:05d}.pt")
    del payload, model, optim
    dist.barrier()
    if rank == 0:
        previous._atomic_json(
            staging / "run_config.json",
            json.loads((output / "run_config.json").read_text()),
        )
        previous._atomic_json(
            staging / "trainer_state.json",
            {
                "format_version": FORMAT_VERSION,
                "schema": SCHEMA,
                "global_step": step,
                "cursor": dataclasses.asdict(cursor),
                "cursor_unit": "context_sorted_response",
                "run_contract_sha256": contract_hash,
                "rank_files": [
                    f"rank-{i:05d}.pt" for i in range(dist.get_world_size())
                ],
                "summary_prefix": summary_prefix,
            },
        )
        os.replace(staging, destination)
    dist.barrier()
    return destination


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-model", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--summary-manifest", type=Path, required=True)
    p.add_argument("--selection-manifest", type=Path, help="Explicit immutable production selection; omitted means full coverage")
    p.add_argument("--streaming-summaries", action="store_true",
                   help="Consume immutable published shards; checkpoint at a missing sorted prefix")
    p.add_argument("--context-order-plan", type=Path, required=True)
    p.add_argument("--memory-budget", type=int, default=4096)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-from", type=Path)
    source.add_argument("--resume-from", type=Path)
    p.add_argument("--repeat-model-export", type=Path, help="Verified model-only export required to change repeat FSDP world size")
    p.add_argument("--gpu-optimizer-states", action="store_true", help="Keep sharded Adam moments on GPU")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--accumulate-tokens", type=int, default=2048)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--keep-last-checkpoints", type=int, default=2)
    p.add_argument("--max-additional-steps", type=int, default=0)
    p.add_argument("--canary-ordinals", default="")
    p.add_argument("--offload-after-tokens", type=int, default=4096)
    p.add_argument("--projection-tokens", type=int, default=64)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.epochs <= 0 or args.accumulate_tokens <= 0 or args.learning_rate <= 0:
        raise ValueError("invalid training budget")
    if min(args.projection_tokens, args.offload_after_tokens, args.keep_last_checkpoints) <= 0:
        raise ValueError("projection/offload/retention budgets must be positive")
    if args.memory_budget < 64 or min(args.warmup_steps, args.max_additional_steps, args.save_interval) < 0:
        raise ValueError("invalid residual budget")
    manifest, database = validate_dataset(args.manifest, args.target_model)
    canary = [int(x) for x in args.canary_ordinals.split(",") if x]
    selection_receipt = None
    selection = canary
    if args.selection_manifest:
        if canary or args.streaming_summaries:
            raise ValueError("production selection cannot be combined with canary/streaming mode")
        selection, selection_receipt = load_selection(args.selection_manifest, manifest, args.summary_manifest)
    order, order_receipt = load_context_order(args.context_order_plan,
        dataset_manifest=manifest, canary_ordinals=selection)
    if args.streaming_summaries:
        if canary:
            raise ValueError("streaming summaries require the complete global curriculum")
        summaries = StreamingSummaryCache(args.summary_manifest, dataset_manifest=manifest,
                                          order=order, order_receipt=order_receipt)
        summary_manifest = summaries.manifest
        available_responses = summaries.available
    else:
        summaries, summary_manifest = open_summary_manifest(args.summary_manifest,
            dataset_manifest=manifest, canary_ordinals=selection)
        available_responses = len(order)
    count, total_tokens, total_responses = database.execute(
        "SELECT COUNT(*), SUM(tokens), SUM(responses) FROM examples"
    ).fetchone()
    if canary and (len(set(canary)) != len(canary) or min(canary) < 0 or max(canary) >= count):
        raise ValueError("invalid canary ordinals")
    if selection_receipt is not None:
        count = total_responses = len(order)
        total_tokens = sum(row[4] for row in order)
        if total_tokens != selection_receipt["selected_target_tokens"]:
            raise ValueError("selection target-token coverage mismatch")
    coverage = summary_manifest["coverage"]
    if coverage["ordinals"] is None and (coverage["responses"] != total_responses or coverage["trajectories"] != count):
        raise ValueError("summary coverage does not include every trajectory/response")
    if not args.streaming_summaries and summaries.db.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] != coverage["responses"]:
        raise ValueError("summary row count changed")
    if args.validate_only:
        print(json.dumps({"validated": True, "trajectories": count, "responses": total_responses,
                          "target_tokens_per_epoch": total_tokens, "summary_coverage": coverage,
                          "memory_budget": args.memory_budget, "target_lora": False,
                          "available_summary_responses": available_responses,
                          "context_curriculum": order_receipt}), flush=True)
        summaries.close()
        database.close()
        return
    if not (args.init_from or args.resume_from):
        raise ValueError("training requires --init-from repeat checkpoint or --resume-from residual checkpoint")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(8)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    world_size = dist.get_world_size()
    if world_size not in (2, 4):
        raise ValueError("requires TP2/FSDP2 or TP4/FSDP4")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    output = args.output_dir.resolve()
    source = (args.resume_from or args.init_from).resolve()
    if args.resume_from and source.parent != output:
        raise ValueError("resume must use the checkpoint's existing output directory")
    source_trainer = json.loads((source / "trainer_state.json").read_text())
    if args.streaming_summaries:
        if args.resume_from:
            summaries.verify_checkpoint_prefix(source_trainer.get("summary_prefix"))
            start_position = source_trainer["cursor"]["event_index"]
        else:
            start_position = 0
        if start_position >= available_responses:
            raise ValueError("no new sorted summaries are ready; generate the next shard before loading GPUs")
        validate_boundary(order, start_position, available_responses, args.accumulate_tokens)
    source_contract = previous._source_run_contract(source)
    source_hash = previous._canonical_json_sha256(source_contract)
    if source_hash != source_trainer["run_contract_sha256"]:
        raise ValueError("source checkpoint/config identity mismatch")
    if args.init_from and source_trainer["schema"] not in {
        previous.CHECKPOINT_SCHEMA, "qwen38_mmfine_full_repeat_reverse_kl_frozen_reader_v1",
        "qwen38_long_document_full_repeat_reverse_kl_frozen_reader_v1",
    }:
        raise ValueError("new residual stage must initialize from a repeat checkpoint (no LoRA)")
    if args.resume_from and source_trainer["schema"] != SCHEMA:
        raise ValueError("resume requires a residual checkpoint")
    weight = (
        source_contract.get("position_weighting")
        if args.resume_from
        else source_contract.get("position_weights")
    )
    # Current MMFine config uses position_weighting; require exact explicit parameters.
    if weight is None:
        weight = source_contract.get("position_weighting")
    if not isinstance(weight, dict):
        raise ValueError("source lacks token-position weighting contract")
    resolved = {k: float(weight[k]) for k in ("alpha", "tau", "power")}
    if resolved != {"alpha": 1.0, "tau": 64.0, "power": 0.5}:
        raise ValueError("unexpected current experiment position weighting")
    repeat_export = None
    if args.repeat_model_export:
        if not args.init_from:
            raise ValueError("repeat export is only for new-stage initialization")
        repeat_export = read_export(args.repeat_model_export, source=source,
            source_trainer=source_trainer, source_hash=source_hash)
    if args.init_from and len(source_trainer["rank_files"]) != world_size and repeat_export is None:
        raise ValueError("world-size migration requires --repeat-model-export")
    contract = {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA,
        "target_model": str(args.target_model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest(args.manifest),
        "database_sha256": manifest["database_sha256"],
        "data_stats": manifest["stats"],
        "context_curriculum": order_receipt,
        "training_selection": selection_receipt,
        "compressor": ({
            **source_contract["compressor"], "architecture": "SummaryResidualCompressor",
            "max_depth": 8, "summary_gate_init": 1e-3,
            "summary_conditioning": "independent_slot_Q_summary_KV_after_source_attention",
            "depth_coordinate": "input_depth_raw_is_zero",
        } if args.init_from else source_contract["compressor"]),
        "target_lora": False, "target_trainable_parameters": 0,
        "parallelism": {"reader_tensor_parallel": world_size, "compressor_fsdp": world_size,
                        "data_parallel_replicas": 1, "same_event_on_all_ranks": True},
        "summary_manifest": str(args.summary_manifest.resolve()),
        "summary_manifest_sha256": digest(args.summary_manifest),
        "summary_cache_sha256": summary_manifest["cache_sha256"],
        "summary_contract": summary_manifest["contract"],
        "summary_delivery": "immutable_append_only_sorted_shards_v1" if args.streaming_summaries else "sealed_full_cache",
        "objective": "full_vocabulary_reverse_KL_student_to_raw_teacher_T1_no_CE",
        "position_weighting": {
            **resolved,
            "formula": "(1+alpha*(1+position/tau)^(-power))/response_mean",
            "coordinate": "original_response_token_offset_no_chunk_reset",
        },
        "residual": {
            "block_tokens": 1024, "compression_ratio": 16, "memory_budget": args.memory_budget,
            "policy": "budget_driven_oldest_adjacent_equal_depth_pyramid",
            "summary_version": "full_causal_history_rebuild_every_response_same_summary_all_levels",
            "gradient": "detach_all_parent_inputs_exact_local_VJP_every_retained_frontier_node",
            "raw_prefix": "native_system_and_tools", "window_tokens": None,
            "teacher": "uncompressed_history_plus_correct_answer_prefix",
            "student": "summary_plus_soft_memory_plus_same_correct_answer_prefix",
            "projection_chunk_tokens": args.projection_tokens,
            "offload_after_tokens": args.offload_after_tokens,
            "coverage": (manifest["coverage"] if "derivation" in manifest else
                         "every_response_every_token_once_per_epoch_no_round_filter"),
            "data_derivation": manifest.get("derivation"),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "accumulate_min_tokens": args.accumulate_tokens,
            "update_boundary": "whole_response_only",
            "normalization": "scored_token_mean",
            "moment_residency": "GPU_resident_sharded" if args.gpu_optimizer_states else "CPU_between_updates",
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "canary_ordinals": canary,
        "implementation": implementation_receipts(),
        "initialization": (
            source_contract["initialization"]
            if args.resume_from
            else {
                "checkpoint": str(source),
                "run_contract_sha256": source_hash,
                "global_step": 0,
                "repeat_global_step": source_trainer["global_step"],
                "model_export": repeat_export,
                "weights_optimizer_rng": "repeat_backbone_weights_only_fresh_Adam_and_seeded_new_parameters",
                "data_cursor": "reset_for_new_objective",
            }
        ),
    }
    contract_hash = previous._canonical_json_sha256(contract)
    if args.resume_from and contract != source_contract:
        raise ValueError("exact residual resume contract changed")
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume_from:
            raise FileExistsError("new-stage output is not empty")
        output.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "phase": "loading_model",
                    "rows": count,
                    "responses": total_responses,
                    "target_tokens_per_epoch": total_tokens,
                    "source_checkpoint": str(source),
                }
            ),
            flush=True,
        )
    model = load_qwen38_multimodal_model(
        str(args.target_model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        tp_plan="auto",
    )
    target = FrozenQwen38MultimodalTarget(model, gradient_checkpointing=True)
    base = SummaryResidualCompressor.canonical_qwen36_27b(
        output_embedding_rms=source_contract["compressor"]["output_embedding_rms"],
        compression_ratio=16,
        local_window=2048,
        gradient_checkpointing=True,
    ).float()
    if base.export_config() != contract["compressor"]:
        raise ValueError("residual architecture does not match the repeat/resume contract")
    if repeat_export is not None:
        initialize_base_from_export(base, args.repeat_model_export, repeat_export)
    compressor = FSDP(
        base,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )
    compressor.train()
    optimizer = torch.optim.AdamW(
        compressor.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )
    residency = GPUMomentResidency() if args.gpu_optimizer_states else previous.LowPeakAdamMomentResidency()
    if args.resume_from:
        step, cursor = previous._load_checkpoint(
            source, compressor=compressor, optimizer=optimizer, residency=residency, rank=rank,
            checkpoint_format_version=FORMAT_VERSION, checkpoint_schema=SCHEMA,
            checkpoint_run_contract_sha256=source_hash)
    else:
        if repeat_export is None:
            initialize_from_repeat(source, compressor=compressor, rank=rank,
                                   source_trainer=source_trainer, source_hash=source_hash)
        step, cursor = 0, previous.QueueCursor(0, 0, 0, 0)
    validate_response_cursor(cursor, epochs=args.epochs, responses=len(order))
    initial_step = step
    optimizer.zero_grad(set_to_none=True)
    writer = None
    if rank == 0:
        previous._atomic_json(
            output / "run_config.json",
            {
                **contract,
                "run_contract_sha256": contract_hash,
                "output_dir": str(output),
                "tensorboard_dir": str(output / "tensorboard"),
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "init_compressor": str(args.init_from) if args.init_from else None,
            },
        )
        writer = SummaryWriter(
            str(output / "tensorboard"),
            purge_step=step + 1 if args.resume_from else None,
        )
        if not args.resume_from:
            writer.file_writer.add_event(
                Event(
                    wall_time=time.time(),
                    step=step,
                    session_log=SessionLog(status=SessionLog.START),
                )
            )
        writer.flush()
    started = time.monotonic()
    pending = {
        "loss_sum": 0.0,
        "correct_sum": 0.0,
        "tokens": 0,
        "reader_forwards": 0,
        "teacher_forwards": 0,
        "summary_forwards": 0,
        "parent_forwards": 0,
        "compressor_forwards": 0,
        "frontier_vjps": 0,
        "source_tokens": 0,
        "covered_source_tokens": 0,
        "max_depth": 0,
        "responses": 0,
        "trajectories": 0,
        "xhigh_trajectories": 0,
        "coding_trajectories": 0,
        "max_soft_tokens": 0,
        "max_teacher_tokens": 0,
        "max_response_tokens": 0,
        "min_context_tokens": 0,
        "max_context_tokens": 0,
    }
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGUSR1, request_stop)
    last_saved = -1
    last_summary = None

    def update(force_save: bool = False):
        nonlocal step, pending, last_saved, last_summary
        if not pending["tokens"]:
            return
        previous._normalize_gradients(compressor.parameters(), pending["tokens"])
        grad_norm = float(compressor.clip_grad_norm_(1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("nonfinite compressor gradient norm")
        lr = previous._learning_rate(step, base=args.learning_rate, warmup=args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if args.gpu_optimizer_states:
            residency.restore(optimizer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            previous._step_optimizer(optimizer, residency)
        step += 1
        if not previous._parameters_finite(compressor):
            raise FloatingPointError("nonfinite compressor parameters")
        last_summary = {
            "global_step": step,
            "stage_step": step - contract["initialization"]["global_step"],
            "loss": pending["loss_sum"] / pending["tokens"],
            "top1_agreement": pending["correct_sum"] / pending["tokens"],
            "target_lora": False,
            "target_trainable_parameters": 0,
            "reverse_kl": pending["loss_sum"] / pending["tokens"],
            "grad_norm": grad_norm,
            "learning_rate": lr,
            "elapsed_seconds": time.monotonic() - started,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "cursor": dataclasses.asdict(cursor),
            "moment_residency": residency.receipt(),
            **{k: v for k, v in pending.items() if not k.endswith("_sum")},
        }
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(last_summary, allow_nan=False) + "\n")
            if writer:
                previous._write_tensorboard_summary(
                    writer, last_summary, global_step=step
                )
            print(json.dumps(last_summary, allow_nan=False), flush=True)
        if force_save or (args.save_interval and step % args.save_interval == 0):
            save_checkpoint(
                output,
                compressor=compressor,
                optimizer=optimizer,
                residency=residency,
                step=step,
                cursor=cursor,
                contract_hash=contract_hash,
                rank=rank,
                summary_prefix=summaries.receipt() if args.streaming_summaries else None,
            )
            last_saved = step
            previous._prune_checkpoints(
                output, keep_last=args.keep_last_checkpoints, rank=rank
            )
        pending = {k: 0 for k in pending}

    start_cursor = cursor
    stop = False
    for epoch in range(cursor.epoch, args.epochs):
        first = start_cursor.event_index if epoch == start_cursor.epoch else 0
        stop_flag = torch.zeros((), dtype=torch.long, device=device)
        for position in range(first, available_responses):
            event, corpus, last_in_trajectory = load_ordered_response(database, order[position])
            identities = [None] * world_size
            dist.all_gather_object(identities, (epoch, position, event.key, event.turn, event.history_sha256))
            if any(x != identities[0] for x in identities):
                raise RuntimeError("tensor-parallel ranks received different events")
            residency.prepare_forward(optimizer)
            objective = ResidualObjective(
                target,
                compressor,
                summary_manifest["contract"],
                memory_budget=args.memory_budget,
                **resolved,
                offload_after_tokens=args.offload_after_tokens,
                projection_tokens=args.projection_tokens,
            )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "phase": "response_start",
                            "ordinal": order[position][0],
                            "turn": event.turn,
                            "key": event.key,
                            "epoch": epoch,
                            "position": position,
                            "context_tokens": order[position][2],
                            "response_tokens": len(event.target_ids),
                        }
                    ),
                    flush=True,
                )
            response_started = time.monotonic()
            metrics = objective.backward_response(event, summaries.get(event))
            if rank == 0:
                print(json.dumps(dict(phase="response_backward_complete", position=position,
                    elapsed_seconds=time.monotonic()-response_started, **metrics)), flush=True)
            for key, value in metrics.items():
                pending[key] = (
                    max(pending[key], value)
                    if key.startswith("max_")
                    else pending[key] + value
                )
            # Count each trajectory once when its final response has been scored.
            pending["trajectories"] += int(last_in_trajectory)
            pending[f"{corpus}_trajectories"] += int(last_in_trajectory)
            context_tokens = order[position][2]
            pending["min_context_tokens"] = min(pending["min_context_tokens"] or context_tokens, context_tokens)
            pending["max_context_tokens"] = max(pending["max_context_tokens"], context_tokens)
            cursor = (
                previous.QueueCursor(epoch + 1, 0, 0, 0)
                if position + 1 == len(order)
                else previous.QueueCursor(epoch, 0, position + 1, 0)
            )
            stop_flag = torch.tensor(
                int(stopping or (output / "STOP_AFTER_RESPONSE").exists()
                    or (available_responses < len(order) and position + 1 == available_responses)),
                device=device,
            )
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            stop = bool(stop_flag) or position + 1 == len(order)
            if pending["tokens"] >= args.accumulate_tokens or stop:
                update(force_save=stop)
                if (
                    args.max_additional_steps
                    and step - initial_step >= args.max_additional_steps
                ):
                    stop = True
                    break
            if bool(stop_flag):
                break
        if bool(stop_flag) or (
            args.max_additional_steps
            and step - initial_step >= args.max_additional_steps
        ):
            break
        cursor = previous.QueueCursor(epoch + 1, 0, 0, 0)
        stop = False
    if step > initial_step and last_saved != step:
        save_checkpoint(
            output,
            compressor=compressor,
            optimizer=optimizer,
            residency=residency,
            step=step,
            cursor=cursor,
            contract_hash=contract_hash,
            rank=rank,
            summary_prefix=summaries.receipt() if args.streaming_summaries else None,
        )
    if rank == 0:
        previous._atomic_json(
            output / "completion.json",
            {
                "global_step": step,
                "cursor": dataclasses.asdict(cursor),
                "all_epochs_complete": cursor.epoch >= args.epochs,
                "awaiting_summaries": available_responses < len(order) and cursor.event_index == available_responses,
                "available_summary_responses": available_responses,
                "last_summary": last_summary,
                "checkpoint": str(output / f"checkpoint-{step:08d}"),
            },
        )
        if writer:
            writer.close()
    summaries.close()
    database.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
