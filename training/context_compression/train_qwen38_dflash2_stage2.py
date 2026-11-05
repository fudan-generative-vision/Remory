"""Offline Stage II from pretrained DFlash2 or Stage I, with warmup and cosine decay.

Each global batch has at least four responses. Every example appears once,
gradients are summed across replicas and divided by the same global token count.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
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
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
)
from torch.utils.tensorboard import SummaryWriter

from context_compression import train_qwen38_mmfine_joint as old
from context_compression.prepare_qwen38_residual_summaries import validate_dataset
from context_compression.prepare_qwen38_summary_selection import load_selection
from context_compression.qwen38_allturn_data import digest
from context_compression.qwen38_multimodal_target import FrozenQwen38MultimodalTarget, load_qwen38_multimodal_model
from context_compression.qwen38_residual_curriculum import load_context_order, load_ordered_response, validate_response_cursor
from context_compression.qwen38_residual_data import open_summary_manifest
from context_compression.qwen38_dflash2_residual import build_residual, load_pretrained_residual, load_stage1_residual
from context_compression.qwen38_residual_training import ResidualObjective
from context_compression.train_qwen38_allturn_residual_kl import implementation_receipts
from context_compression.stage2_batch_order import global_batches, epoch_batch_plan, resume_batch_index
from context_compression.stage2_memory_policy import MemoryBudgetResidualObjective

SCHEMA = "qwen38_pretrained_dflash2_offline_stage2_shuffled_v2"


def learning_rate(step, *, peak, warmup, total, minimum):
    """step is completed updates; update warmup reaches peak, final update minimum."""
    if not 0 <= step < total or not 0 <= warmup < total or not 0 <= minimum <= peak:
        raise ValueError("invalid warmup/cosine schedule coordinate")
    update = step + 1
    if warmup and update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (total - warmup)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def parameter_fingerprint(model):
    """Cheap rank consistency check before FSDP singleton-group wrapping."""
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(f"{name}:{tuple(value.shape)}".encode())
        flat = value.detach().reshape(-1)
        h.update(torch.cat((flat[:64], flat[-64:])).float().cpu().numpy().tobytes())
    return h.hexdigest()


def read_fast_checkpoint(path, world):
    path = Path(path).resolve()
    previous = json.loads((path / "run_config.json").read_text())
    trainer = json.loads((path / "trainer_state.json").read_text())
    contract = {k: v for k, v in previous.items()
                if k not in ("run_contract_sha256", "output_dir", "resume_from")}
    checksum = old._canonical_json_sha256(contract)
    if (trainer["schema"] != SCHEMA or previous["schema"] != SCHEMA
            or trainer["world_size"] != world
            or checksum != trainer["run_contract_sha256"]
            or checksum != previous["run_contract_sha256"]
            or previous["benchmark_ordinals"]):
        raise ValueError("unsupported or inconsistent fast checkpoint")
    names = ["model.pt", "optimizer.pt", "run_config.json", "trainer_state.json",
             *[f"rng-{rank}.pt" for rank in range(world)]]
    if any(not (path / name).is_file() or not (path / name).stat().st_size for name in names):
        raise ValueError("incomplete fast checkpoint")
    return previous, trainer, names


def assign_batch(order, start, end, replicas):
    """Deterministic longest-estimated-job-first, retaining each row exactly once."""
    jobs, loads = [[] for _ in range(replicas)], [0] * replicas
    for position in sorted(range(start, end), key=lambda i: (-(order[i][2] + 2 * order[i][4]), i)):
        rank = min(range(replicas), key=lambda i: (loads[i], i))
        jobs[rank].append(position)
        loads[rank] += order[position][2] + 2 * order[position][4]
    return [sorted(x) for x in jobs]


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--pretrained-init", type=Path)
    source.add_argument("--resume-from", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--context-order-plan", type=Path, required=True)
    p.add_argument("--mode", choices=("tp", "dp"), default="dp")
    p.add_argument("--feature-residency", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--node-batch-size", type=int, default=8)
    p.add_argument("--projection-tokens", type=int, default=512)
    p.add_argument("--compressor-checkpointing", action="store_true")
    p.add_argument("--reader-checkpoint-after-tokens", type=int, default=0,
                   help="0 keeps reader checkpointing; positive threshold retains activations for shorter student sequences")
    p.add_argument("--reader-layer-token-budget", type=int, default=0,
                   help="Keep up to this many reader layer-token activations; checkpoint the remaining layers")
    p.add_argument("--reader-retain-min-tokens", type=int, default=0)
    p.add_argument("--short-response-node-batch-size", type=int, default=32)
    p.add_argument("--allow-execution-change", action="store_true",
                   help="Explicit one-time migration of checked memory/compute settings; preserve data, model, optimizer, RNG and cursor")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--keep-last-checkpoints", type=int, default=3)
    p.add_argument("--max-additional-steps", type=int, default=0)
    p.add_argument("--benchmark-ordinals", default="")
    p.add_argument("--benchmark-passes", type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    if min(args.node_batch_size, args.projection_tokens, args.benchmark_passes) < 1:
        raise ValueError("execution sizes must be positive")
    if args.reader_checkpoint_after_tokens < 0:
        raise ValueError("reader checkpoint threshold must be nonnegative")
    if args.reader_layer_token_budget < 0 or (args.reader_layer_token_budget and args.reader_checkpoint_after_tokens):
        raise ValueError("reader layer budget is nonnegative and exclusive with whole-reader threshold")
    if args.reader_retain_min_tokens < 0 or args.short_response_node_batch_size < 1:
        raise ValueError("invalid conditional memory policy")
    if args.allow_execution_change and not args.resume_from:
        raise ValueError("execution migration requires a complete existing checkpoint")
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    output = args.output_dir.resolve()
    if args.pretrained_init:
        spec = json.loads(args.pretrained_init.read_text())
        source, receipt = spec["source_contract"], spec["initialization_receipt"]
        state_dir = None
        step, cursor = 0, old.QueueCursor(0, 0, 0, 0)
    else:
        state_dir = args.resume_from.resolve()
        if state_dir.parent != output:
            raise ValueError("resume must remain in its existing run directory")
        previous, trainer, names = read_fast_checkpoint(state_dir, world)
        source, receipt = previous["source_contract"], previous["migration"]
        step, cursor = trainer["global_step"], old.QueueCursor(**trainer["cursor"])
    weights_contract = source["pretrained"]
    stage1_contract = source.get("stage1_checkpoint")
    if rank == 0:
        for name, expected in weights_contract["files"].items():
            if digest(Path(weights_contract["directory"]) / name) != expected:
                raise ValueError(f"pretrained artifact changed: {name}")
        if stage1_contract:
            for name, expected in stage1_contract["files"].items():
                if digest(Path(stage1_contract["directory"]) / name) != expected:
                    raise ValueError(f"Stage I artifact changed: {name}")
    dist.barrier()
    if receipt["source_world_size"] != world:
        raise ValueError("this migration preserves one RNG stream per original rank")
    random.seed(source["seed"])
    torch.manual_seed(source["seed"])
    torch.cuda.manual_seed(source["seed"])
    for key in ("manifest", "summary_manifest"):
        if digest(Path(source[key])) != source[key + "_sha256"]:
            raise ValueError(f"sealed data contract changed: {key}")
    manifest, database = validate_dataset(Path(source["manifest"]), Path(source["target_model"]))
    if manifest["database_sha256"] != source["database_sha256"]:
        raise ValueError("source database identity changed")
    selected, selection_receipt = load_selection(source["training_selection"]["manifest"], manifest, source["summary_manifest"])
    if selection_receipt != source["training_selection"]:
        raise ValueError("fixed training cohort changed")
    order, order_receipt = load_context_order(args.context_order_plan,
        dataset_manifest=manifest, canary_ordinals=selected)
    if order_receipt != source["context_curriculum"]:
        raise ValueError("curriculum changed")
    if source.get("batch_order") != dict(schema="fixed_length_bucket_batch_shuffle_v1",
        seed=source["seed"], epoch_seed="seed_plus_zero_based_epoch",
        base_batches="immutable_context_ascending_whole_response_batches",
        permutation="python_random_Random_shuffle", cursor="response_offset_in_epoch_permuted_batch_plan",
        rebatch_after_shuffle=False):
        raise ValueError("unexpected batch shuffle contract")
    validate_response_cursor(cursor, epochs=source["epochs"], responses=len(order))
    summaries, summary_manifest = open_summary_manifest(source["summary_manifest"], dataset_manifest=manifest, canary_ordinals=selected)
    if (summary_manifest["cache_sha256"] != source["summary_cache_sha256"]
            or summary_manifest["contract"] != source["summary_contract"]):
        raise ValueError("summary identity changed")
    planned_per_epoch = sum(1 for _ in global_batches(order, 0, source["optimizer"]["accumulate_min_tokens"]))
    if planned_per_epoch * source["epochs"] != source["optimizer"]["total_steps"]:
        raise ValueError("schedule total does not match exact whole-response epoch batches")
    if cursor.epoch >= source["epochs"]:
        raise ValueError("all five epochs already complete")
    bench = [int(x) for x in args.benchmark_ordinals.split(",") if x]
    if bench:
        wanted = set(bench)
        if len(wanted) != len(bench) or not wanted.issubset(set(selected)):
            raise ValueError("benchmark must use distinct selected events")
        order = [row for row in order if row[0] in wanted]
    execution = dict(mode=args.mode, world_size=world, reader_tensor_parallel=world if args.mode == "tp" else 1,
        data_parallel_replicas=1 if args.mode == "tp" else world,
        feature_residency=args.feature_residency, node_batch_size=args.node_batch_size,
        projection_tokens=args.projection_tokens, compressor_checkpointing=args.compressor_checkpointing,
        reader_checkpoint_after_tokens=args.reader_checkpoint_after_tokens, fused_adam=args.fused_adam,
        reader_layer_token_budget=args.reader_layer_token_budget, memory_metrics_window="per_update",
        reader_retain_min_tokens=args.reader_retain_min_tokens, short_response_node_batch_size=args.short_response_node_batch_size,
        global_batch="whole_response_token_threshold_minimum_four_responses_merge_short_tail", assignment="greedy_context_plus_twice_target",
        reduction="sum_replica_gradients_then_divide_by_global_tokens",
        epoch_order="fixed_length_similar_batches_shuffled_each_epoch",
        preserve_epoch_checkpoints=True)
    implementations = implementation_receipts()
    implementations[Path(__file__).name] = digest(Path(__file__))
    implementations["export_qwen38_residual_training_state.py"] = digest(Path(__file__).with_name("export_qwen38_residual_training_state.py"))
    for name in ("qwen38_dflash2_residual.py", "qwen38_dflash2_compressor.py", "dflash2_grouped_conv.py", "stage2_batch_order.py", "stage2_memory_policy.py"):
        implementations[name] = digest(Path(__file__).with_name(name))
    contract = dict(schema=SCHEMA, source_contract=source, migration=receipt, execution=execution,
                    implementation=implementations, benchmark_ordinals=bench, epochs=source["epochs"])
    contract_hash = old._canonical_json_sha256(contract)
    rank_contracts = [None] * world
    dist.all_gather_object(rank_contracts, contract_hash)
    if len(set(rank_contracts)) != 1:
        raise ValueError("ranks disagree on data or execution contract")
    if args.resume_from:
        if contract_hash != trainer["run_contract_sha256"] or contract_hash != previous["run_contract_sha256"]:
            if not args.allow_execution_change:
                raise ValueError("fast resume contract changed; explicit execution migration is required")
            for key in ("schema", "source_contract", "migration", "benchmark_ordinals", "epochs"):
                if contract[key] != previous[key]:
                    raise ValueError(f"execution migration changed scientific contract: {key}")
            changed_execution = {k for k in set(previous["execution"]) | set(execution)
                                 if previous["execution"].get(k) != execution.get(k)}
            if not changed_execution <= {"node_batch_size", "projection_tokens", "reader_layer_token_budget", "memory_metrics_window", "reader_retain_min_tokens", "short_response_node_batch_size"}:
                raise ValueError("execution migration changed unsupported execution fields")
            changed_implementation = {k for k in set(previous["implementation"]) | set(implementations)
                                      if previous["implementation"].get(k) != implementations.get(k)}
            if not changed_implementation <= {Path(__file__).name, "stage2_memory_policy.py"}:
                raise ValueError("execution migration changed objective/model/data implementations")
            if rank == 0:
                old._atomic_json(output / f"execution-migration-step{step:08d}.json", dict(
                    checkpoint=str(state_dir), global_step=step, cursor=dataclasses.asdict(cursor),
                    old_contract_sha256=previous["run_contract_sha256"], new_contract_sha256=contract_hash,
                    previous_execution=previous["execution"], execution=execution,
                    changed_execution=sorted(changed_execution), changed_implementation=sorted(changed_implementation),
                    source_contract_unchanged=True, full_optimizer_and_rng_restore_required=True))
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume_from:
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(dict(phase="loading_model", execution=execution, source_step=step, cursor=dataclasses.asdict(cursor))), flush=True)
    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    kwargs.update({"tp_plan": "auto"} if args.mode == "tp" else {"device_map": {"": device}})
    target = FrozenQwen38MultimodalTarget(load_qwen38_multimodal_model(source["target_model"], **kwargs), gradient_checkpointing=True,
        target_layer_ids=source["compressor"]["target_layer_ids"])
    target_trainable = sum(p.numel() for p in target.model.parameters() if p.requires_grad)
    if target_trainable:
        raise ValueError("the Qwen target must remain fully frozen")
    if args.pretrained_init:
        # Model loading must not change the Compressor's initialization stream.
        random.seed(source["seed"])
        torch.manual_seed(source["seed"])
        torch.cuda.manual_seed(source["seed"])
    base = build_residual(Path(weights_contract["directory"]) / "config.json",
        output_embedding_rms=source["compressor"]["output_embedding_rms"],
        gradient_checkpointing=args.compressor_checkpointing).float()
    expected_config = {**source["compressor"], "gradient_checkpointing": args.compressor_checkpointing}
    if base.export_config() != expected_config:
        raise ValueError("compressor architecture changed")
    if state_dir is not None:
        weights = torch.load(state_dir / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        base.load_state_dict(weights, strict=True)
        del weights
    elif stage1_contract:
        initialization = load_stage1_residual(base, stage1_contract["directory"])
        if rank == 0:
            old._atomic_json(output / "stage1-load-audit.json", initialization)
    else:
        initialization = load_pretrained_residual(base,
            Path(weights_contract["directory"]) / "model.safetensors", source["target_model"])
        if rank == 0:
            old._atomic_json(output / "pretrained-load-audit.json", initialization)
    initial_fingerprint = parameter_fingerprint(base)
    fingerprints = [None] * world
    dist.all_gather_object(fingerprints, initial_fingerprint)
    if len(set(fingerprints)) != 1:
        raise ValueError("Compressor replicas differ before training")
    compressor_numel = sum(p.numel() for p in base.parameters())
    group = singleton_group(rank, world) if args.mode == "dp" else None
    compressor = FSDP(base, device_id=device, process_group=group,
        sharding_strategy=ShardingStrategy.NO_SHARD if args.mode == "dp" else ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                        buffer_dtype=torch.bfloat16, cast_forward_inputs=True),
        use_orig_params=True, sync_module_states=True, limit_all_gathers=True)
    compressor.train()
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=source["optimizer"]["learning_rate"],
        betas=tuple(source["optimizer"]["betas"]), eps=source["optimizer"]["eps"],
        weight_decay=source["optimizer"]["weight_decay"], foreach=False, fused=args.fused_adam)
    if state_dir is not None:
        load_full_optimizer(compressor, optimizer, state_dir / "optimizer.pt", fused=args.fused_adam)
        restore_rng(state_dir / f"rng-{rank}.pt")
    elif optimizer.state:
        raise ValueError("pretrained initialization must have an empty optimizer state")
    objective = MemoryBudgetResidualObjective(target, compressor, summary_manifest["contract"],
        memory_budget=source["residual"]["memory_budget"],
        alpha=source["position_weighting"]["alpha"], tau=source["position_weighting"]["tau"],
        power=source["position_weighting"]["power"], projection_tokens=args.projection_tokens,
        offload_after_tokens=131072, feature_residency=args.feature_residency, node_batch_size=args.node_batch_size,
        reader_checkpoint_after_tokens=args.reader_checkpoint_after_tokens,
        reader_layer_token_budget=args.reader_layer_token_budget,
        reader_retain_min_tokens=args.reader_retain_min_tokens,
        short_response_node_batch_size=args.short_response_node_batch_size)
    if rank == 0:
        initialization_status = dict(phase="training_ready", global_step=step,
            initialization=source["initialization"], state_loaded_from=str(state_dir) if state_dir else (stage1_contract or weights_contract)["directory"],
            planned_updates=source["optimizer"]["total_steps"], updates_per_epoch=planned_per_epoch,
            compressor_parameters=compressor_numel, target_trainable_parameters=target_trainable,
            sampled_parameter_sha256=initial_fingerprint, replicas_identical=True,
            optimizer_state_entries=len(optimizer.state), cursor=dataclasses.asdict(cursor))
        print(json.dumps(initialization_status), flush=True)
        old._atomic_json(output / f"initialization-step{step:08d}.json", initialization_status)
        old._atomic_json(output / "run_config.json", {**contract, "run_contract_sha256": contract_hash,
            "output_dir": str(output), "resume_from": str(args.resume_from) if args.resume_from else None})
        if args.resume_from and (output / "completion.json").exists():
            (output / "completion.json").rename(output / f"completion-before-resume-step{step:08d}.json")
    dist.barrier()
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=step + 1 if args.resume_from else None) if rank == 0 and not bench else None
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGUSR1, request_stop)
    initial_step, last_saved, began = step, -1, time.monotonic()
    stop = torch.tensor(0, device=device)
    benchmark_rows = []
    epochs = range(args.benchmark_passes) if bench else range(cursor.epoch, source["epochs"])
    for epoch in epochs:
        first = 0 if bench or epoch != cursor.epoch else cursor.event_index
        epoch_order, batches, epoch_receipt = epoch_batch_plan(order,
            threshold=source["optimizer"]["accumulate_min_tokens"], epoch=epoch, seed=source["batch_order"]["seed"])
        first_batch = resume_batch_index(batches, first)
        if not bench and step != epoch * planned_per_epoch + first_batch:
            raise ValueError("checkpoint step does not match shuffled epoch cursor")
        if rank == 0:
            receipt_path = output / f"epoch-order-{epoch:03d}.json"
            if receipt_path.exists() and json.loads(receipt_path.read_text()) != epoch_receipt:
                raise ValueError("epoch batch permutation changed on resume")
            old._atomic_json(receipt_path, epoch_receipt)
        for batch_index in range(first_batch, len(batches)):
            start, end, token_count, base_batch_id = batches[batch_index]
            assignments = assign_batch(epoch_order, start, end, world) if args.mode == "dp" else [list(range(start, end))] * world
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began_batch = time.monotonic()
            local_loss = local_correct = local_tokens = 0
            local_max_depth = 0
            for position in assignments[rank]:
                event, _, _ = load_ordered_response(database, epoch_order[position])
                before = time.monotonic()
                metrics = objective.backward_response(event, summaries.get(event))
                local_loss += metrics["loss_sum"]
                local_correct += metrics["correct_sum"]
                local_tokens += metrics["tokens"]
                local_max_depth = max(local_max_depth, metrics["max_depth"])
                with (output / f"events-rank{rank}.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(epoch=epoch, position=position, ordinal=epoch_order[position][0],
                        base_batch_id=base_batch_id, batch_index=batch_index,
                        context_tokens=epoch_order[position][2], elapsed_seconds=time.monotonic() - before, **metrics)) + "\n")
            if args.mode == "dp":
                sum_replica_gradients(compressor.parameters())
            old._normalize_gradients(compressor.parameters(), token_count)
            if any(p.requires_grad or p.grad is not None for p in target.model.parameters()):
                raise RuntimeError("frozen target received a gradient or became trainable")
            norm = float(compressor.clip_grad_norm_(1.0))
            if not math.isfinite(norm):
                raise FloatingPointError("invalid compressor gradient norm")
            lr = learning_rate(step, peak=source["optimizer"]["learning_rate"],
                warmup=source["optimizer"]["warmup_steps"], total=source["optimizer"]["total_steps"],
                minimum=source["optimizer"]["minimum_learning_rate"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            if not bench:
                optimizer.step()
                step += 1
                if not old._parameters_finite(compressor):
                    raise FloatingPointError("nonfinite compressor weights")
            totals = torch.tensor([local_loss, local_correct, local_tokens], device=device, dtype=torch.float64)
            dist.all_reduce(totals)
            if args.mode == "tp":
                totals /= world
            if int(totals[2]) != token_count:
                raise ValueError("global token coverage mismatch")
            torch.cuda.synchronize()
            elapsed = torch.tensor([time.monotonic() - began_batch, torch.cuda.max_memory_allocated() / 2**20, local_max_depth], device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            cursor = old.QueueCursor(epoch + 1, 0, 0, 0) if end == len(order) else old.QueueCursor(epoch, 0, end, 0)
            row = dict(global_step=step, epoch=epoch, start_position=start, end_position=end,
                batch_index=batch_index, base_batch_id=base_batch_id,
                responses=end - start, tokens=token_count, loss=float(totals[0] / token_count),
                top1_agreement=float(totals[1] / token_count), grad_norm=norm, learning_rate=lr,
                elapsed_seconds=time.monotonic() - began, batch_seconds=float(elapsed[0]),
                gpu_peak_allocated_mib=float(elapsed[1]), max_depth=int(elapsed[2]),
                context_min=min(x[2] for x in epoch_order[start:end]), context_max=max(x[2] for x in epoch_order[start:end]),
                assignments=assignments, target_trainable_parameters=0, cursor=dataclasses.asdict(cursor))
            row["tokens_per_second"] = token_count / float(elapsed[0])
            reserved = torch.tensor([torch.cuda.memory_reserved() / 2**20, torch.cuda.max_memory_reserved() / 2**20], device=device)
            dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
            row["gpu_reserved_mib"], row["gpu_peak_reserved_mib"] = map(float, reserved.tolist())
            row["reader_layer_token_budget"] = args.reader_layer_token_budget
            row["epoch_progress"] = epoch + end / len(order)
            if rank == 0:
                with (output / ("benchmark.jsonl" if bench else "metrics.jsonl")).open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
                if writer:
                    for key in ("loss", "top1_agreement", "grad_norm", "learning_rate", "batch_seconds", "gpu_peak_allocated_mib", "gpu_reserved_mib", "gpu_peak_reserved_mib", "tokens_per_second", "epoch_progress", "context_min", "context_max", "max_depth", "base_batch_id"):
                        writer.add_scalar(key, row[key], step)
            if bench:
                benchmark_rows.append(row)
                continue
            stop = torch.tensor(int(stopping or (output / "STOP_AFTER_RESPONSE").exists()
                or (args.max_additional_steps and step - initial_step >= args.max_additional_steps)), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop) or end == len(order) or (args.save_interval and step % args.save_interval == 0):
                checkpoint = save_full_checkpoint(output, compressor, optimizer, mode=args.mode, rank=rank, step=step,
                                                  cursor=cursor, contract_hash=contract_hash)
                if end == len(order) and rank == 0:
                    # Hard links keep epoch-end states available after rolling pruning.
                    archive = output / "epoch-end-checkpoints" / checkpoint.name
                    archive.mkdir(parents=True, exist_ok=False)
                    for artifact in checkpoint.iterdir():
                        if artifact.is_file():
                            os.link(artifact, archive / artifact.name)
                dist.barrier()
                last_saved = step
                old._prune_checkpoints(output, keep_last=args.keep_last_checkpoints, rank=rank)
            if bool(stop):
                break
        if not bench and bool(stop):
            break
    if rank == 0:
        if bench:
            timed = [row for row in benchmark_rows if row["epoch"] == args.benchmark_passes - 1]
            old._atomic_json(output / "benchmark-result.json", dict(execution=execution, ordinals=bench,
                passes=args.benchmark_passes, final_pass_seconds=sum(x["batch_seconds"] for x in timed),
                final_pass_responses=sum(x["responses"] for x in timed),
                final_pass_target_tokens=sum(x["tokens"] for x in timed),
                peak_allocated_mib=max(x["gpu_peak_allocated_mib"] for x in timed), rows=timed))
        else:
            old._atomic_json(output / "completion.json", dict(global_step=step, cursor=dataclasses.asdict(cursor),
                all_epochs_complete=cursor.epoch >= source["epochs"], stopped_on_request=bool(stop), checkpoint=str(output / f"checkpoint-{last_saved:08d}")))
        if writer:
            writer.close()
    summaries.close()
    database.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
