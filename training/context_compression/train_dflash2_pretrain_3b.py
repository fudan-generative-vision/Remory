"""Large-corpus Stage I: direct pretrained DFlash2, one epoch, warmup/cosine, four GPUs."""
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
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.tensorboard import SummaryWriter

from context_compression import train_qwen38_mmfine_joint as old
from context_compression.dflash2_parallel import (
    SCHEMA, parameter_fingerprint, singleton_group, load_full_optimizer,
    restore_rng, save_full_checkpoint, sum_replica_gradients, full_state_context,
)
from context_compression.qwen38_dflash2_compressor import build_compressor, load_pretrained_backbone, load_target_mask_embedding
from context_compression.qwen38_long_repeat_data import load_dataset, load_example, training_order, bucket
from context_compression.qwen38_long_repeat_kl import full_repeat_kl, render_full_repeat
from context_compression.qwen38_multimodal_target import (
    FrozenQwen38MultimodalTarget, load_qwen38_multimodal_model, load_qwen38_processor,
)


def learning_rate(step, *, peak, warmup, total, minimum):
    if not 0 <= step < total or not 0 <= warmup < total or not 0 <= minimum <= peak:
        raise ValueError('invalid warmup/cosine schedule')
    update = step + 1
    if warmup and update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (total - warmup)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def assign_batch(order, counts, start, end, world):
    """Keep global batch membership; balance estimated long-sequence work across replicas."""
    assignments, loads = [[] for _ in range(world)], [0.0] * world
    def cost(position):
        n = counts[order[position]]
        return n * (1.0 + n / 65536.0)
    for position in sorted(range(start, end), key=lambda i: (-cost(i), i)):
        rank = min(range(world), key=lambda i: (loads[i], i))
        assignments[rank].append(position)
        loads[rank] += cost(position)
    return [sorted(positions) for positions in assignments]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--pretrained-init', type=Path)
    src.add_argument('--resume-from', type=Path)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('dp', 'tp'), default='dp')
    p.add_argument('--projection-tokens', type=int, default=1024)
    p.add_argument('--reader-checkpoint-after-tokens', type=int, default=16384)
    p.add_argument('--fused-adam', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-interval', type=int, default=25)
    p.add_argument('--validation-interval', type=int, default=50)
    p.add_argument('--keep-last-checkpoints', type=int, default=3)
    p.add_argument('--max-additional-steps', type=int, default=0)
    p.add_argument('--initial-validation', action='store_true')
    p.add_argument('--canary-lengths', default='')
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--benchmark-passes', type=int, default=3)
    p.add_argument('--benchmark-batches', type=int, default=2)
    p.add_argument('--benchmark-grid', default='',
                   help='comma-separated projection:checkpoint-threshold pairs, fixed weights/no updates')
    return p


def main():
    args = parser().parse_args()
    if args.projection_tokens <= 0 or args.reader_checkpoint_after_tokens < 0:
        raise ValueError('invalid execution budget')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60), device_id=device)
    world = dist.get_world_size()
    if world != 4:
        raise ValueError('this execution requires four GPUs')
    output = args.output_dir.resolve()
    if args.pretrained_init:
        source = json.loads(args.pretrained_init.read_text())
        state_dir = None
        origin = dict(kind='direct_public_pretrained_DFlash2',
                      stage1_pilot_checkpoint_used=False, stage2_checkpoint_used=False)
        step, cursor = 0, old.QueueCursor(0, 0, 0, 0)
        rng_rank = rank
    else:
        state_dir = args.resume_from.resolve()
        if state_dir.parent != output:
            raise ValueError('resume must stay in the same run directory')
        previous = json.loads((state_dir / 'run_config.json').read_text())
        trainer = json.loads((state_dir / 'trainer_state.json').read_text())
        previous_contract = {k:v for k,v in previous.items()
                             if k not in ('run_contract_sha256', 'output_dir', 'resume_from')}
        if (trainer['schema'] != SCHEMA or trainer['world_size'] != world
            or trainer['run_contract_sha256'] != old._canonical_json_sha256(previous_contract)
            or previous['run_contract_sha256'] != trainer['run_contract_sha256']):
            raise ValueError('invalid four-GPU checkpoint contract')
        for name in ['model.pt', 'optimizer.pt', *[f'rng-{i}.pt' for i in range(world)]]:
            if not (state_dir / name).is_file():
                raise ValueError(f'incomplete checkpoint: {name}')
        source, origin = previous['source_contract'], previous['migration']
        step, cursor = trainer['global_step'], old.QueueCursor(**trainer['cursor'])
        rng_rank = rank
    if source['schema'] != 'qwen38_pretrained_dflash2_reconstruction_source_v2':
        raise ValueError('not a pretrained DFlash2 Stage-I run')
    random.seed(source['seed'])
    torch.manual_seed(source['seed'])
    torch.cuda.manual_seed(source['seed'])
    manifest, db = load_dataset(args.manifest, source['target_model'])
    if manifest != source['dataset']:
        raise ValueError('immutable dataset contract changed')
    counts = dict(db.execute('SELECT id, token_count FROM examples'))
    expected_steps = math.ceil(manifest['counts']['train'] / source['optimizer']['grad_accumulation'])
    if source['epochs'] != 1 or source['optimizer']['total_steps'] != expected_steps:
        raise ValueError('one-epoch schedule does not match sealed dataset')
    if cursor.epoch >= source['epochs']:
        raise ValueError('configured epoch already complete')
    if rank == 0:
        p = Path(source['compressor_initialization']['path'])
        for filename, key in [('config.json','config_sha256'),('model.safetensors','weights_sha256')]:
            if old._sha256(p/filename) != source['compressor_initialization'][key]:
                raise ValueError('pretrained artifact changed')
    dist.barrier()
    implementation = {p.name:old._sha256(p) for p in Path(__file__).parent.glob('*.py')}
    execution = dict(mode=args.mode, world_size=world, projection_tokens=args.projection_tokens,
        reader_checkpoint_after_tokens=args.reader_checkpoint_after_tokens, fused_adam=args.fused_adam,
        reduction=('FP32_gradient_sum_divided_by_global_scored_tokens' if args.mode == 'dp'
                   else 'FSDP_mean_of_identical_TP_response_gradients_divided_by_scored_tokens'),
        assignment='within_unchanged_global_batch_longest_estimated_work_first',
        global_documents_per_update=source['optimizer']['grad_accumulation'],
        rng_migration='fresh_seed_or_exact_same_rank_resume', cpu_threads_per_rank=4)
    contract = dict(schema=SCHEMA, source_contract=source, migration=origin,
                    execution=execution, implementation=implementation,
                    benchmark=args.benchmark, benchmark_grid=args.benchmark_grid,
                    validation_interval=args.validation_interval)
    contract_hash = old._canonical_json_sha256(contract)
    if args.resume_from and contract != previous_contract:
        raise ValueError('exact four-GPU resume contract changed')
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume_from:
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(dict(phase='loading_frozen_model', mode=args.mode, world_size=world,
                              global_step=step, cursor=dataclasses.asdict(cursor))), flush=True)
    dist.barrier()
    processor = load_qwen38_processor(source['target_model'])
    kwargs = dict(dtype=torch.bfloat16, attn_implementation='sdpa')
    kwargs.update({'tp_plan':'auto'} if args.mode == 'tp' else {'device_map':{'':device}})
    base = build_compressor(Path(source['compressor_initialization']['path']) / 'config.json',
                            output_embedding_rms=source['compressor']['output_embedding_rms']).float()
    if base.export_config() != source['compressor']:
        raise ValueError('compressor architecture changed')
    if state_dir is not None:
        weights = torch.load(state_dir / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        base.load_state_dict(weights, strict=True)
        del weights
    else:
        initialization = load_pretrained_backbone(base,
            Path(source['compressor_initialization']['path']) / 'model.safetensors')
        mask_embedding = load_target_mask_embedding(source['target_model'], 248070)
        with torch.no_grad():
            base.mask_embedding.copy_(mask_embedding)
        if rank == 0:
            old._atomic_json(output/'pretrained-load-audit.json', dict(**initialization,
                mask_embedding_equal_target=torch.equal(base.mask_embedding, mask_embedding),
                trainable_parameters=sum(p.numel() for p in base.parameters()),
                stage1_pilot_checkpoint_used=False, stage2_checkpoint_used=False))
    parameter_names = set(dict(base.named_parameters()))
    fingerprint = parameter_fingerprint(base)
    fingerprints = [None] * world
    dist.all_gather_object(fingerprints, fingerprint)
    if len(set(fingerprints)) != 1:
        raise ValueError('initial replicas differ')
    target = FrozenQwen38MultimodalTarget(load_qwen38_multimodal_model(source['target_model'], **kwargs),
        target_layer_ids=base.target_layer_ids, gradient_checkpointing=True)
    group = singleton_group(rank, world) if args.mode == 'dp' else None
    compressor = FSDP(base, device_id=device, process_group=group,
        sharding_strategy=ShardingStrategy.NO_SHARD if args.mode == 'dp' else ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16, cast_forward_inputs=True),
        use_orig_params=True, sync_module_states=True, limit_all_gathers=True)
    compressor.train()
    opt = source['optimizer']
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=opt['learning_rate'],
        betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'],
        foreach=False, fused=args.fused_adam)
    if state_dir is not None:
        load_full_optimizer(compressor, optimizer, state_dir / 'optimizer.pt', fused=args.fused_adam)
        restore_rng(state_dir / f'rng-{rng_rank}.pt')
    optimizer.zero_grad(set_to_none=True)
    if any(p.requires_grad or p.grad is not None for p in target.model.parameters()):
        raise ValueError('target must remain frozen')
    optimizer_steps = sorted(set(float(s['step']) for s in optimizer.state.values() if 'step' in s))
    if (step > 0 and (not optimizer_steps or any(x != step for x in optimizer_steps))) or (step == 0 and optimizer.state):
        raise ValueError(f'optimizer update count lost: {optimizer_steps} vs {step}')
    ready = dict(phase='training_ready', global_step=step, cursor=dataclasses.asdict(cursor),
        state_loaded_from=str(state_dir) if state_dir else source['compressor_initialization']['path'], target_trainable_parameters=0,
        optimizer_state_entries=len(optimizer.state), optimizer_steps=optimizer_steps,
        initial_parameter_fingerprint=fingerprint, replicas_identical=True, execution=execution)
    if rank == 0:
        old._atomic_json(output / f'initialization-step{step:08d}.json', ready)
        old._atomic_json(output / 'run_config.json', {**contract, 'run_contract_sha256':contract_hash,
            'output_dir':str(output), 'resume_from':str(args.resume_from) if args.resume_from else None})
        if args.resume_from and (output/'completion.json').exists():
            (output/'completion.json').rename(output/f'completion-before-resume-step{step:08d}.json')
        print(json.dumps(ready), flush=True)
    projection, threshold = args.projection_tokens, args.reader_checkpoint_after_tokens
    checkpoint_enabled = True

    def response(example, *, backward):
        nonlocal checkpoint_enabled
        rendered = render_full_repeat(processor, example['token_ids'])
        n = len(example['token_ids'])
        width = (len(rendered.student_prefix_before_memory_ids)
                 + ((n+1023)//1024)*64 + len(rendered.student_prefix_after_memory_ids)
                 + len(rendered.target_ids))
        enabled = width > threshold
        if enabled != checkpoint_enabled:
            if enabled:
                target.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else:
                target.model.gradient_checkpointing_disable()
            checkpoint_enabled = enabled
        with torch.set_grad_enabled(backward):
            loss, metrics = full_repeat_kl(rendered, target=target, compressor=compressor,
                source_offset=example['token_start'], projection_tokens=projection,
                offload_after_tokens=source['offload_after_tokens'],
                maximum_source_tokens=source['repeat']['source_chunk_tokens'])
        if backward:
            (loss * metrics['tokens']).backward()
        if not math.isfinite(metrics['reverse_kl']):
            raise FloatingPointError('nonfinite reconstruction KL')
        return metrics

    def verify_unchanged_weights(filename):
        if args.mode == 'tp' or rank == 0:
            with full_state_context(compressor):
                unchanged = compressor.state_dict()
            if rank == 0:
                if state_dir is not None:
                    expected = torch.load(state_dir/'model.pt',map_location='cpu',weights_only=True,mmap=True)
                else:
                    from safetensors.torch import load_file
                    expected = load_file(str(Path(source['compressor_initialization']['path'])/'model.safetensors'))
                    expected['mask_embedding'] = load_target_mask_embedding(source['target_model'],248070)
                changed = [name for name in sorted(parameter_names)
                           if not torch.equal(unchanged[name].detach().cpu(),expected[name].float())]
                if changed:
                    raise RuntimeError(f'preflight changed parameters: {changed}')
                old._atomic_json(output/filename,dict(passed=True,
                    exact_equal_parameter_tensors=len(parameter_names),optimizer_updated=False,
                    source=str(state_dir) if state_dir else source['compressor_initialization']['path']))
                del expected
            del unchanged
        dist.barrier()

    canary_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate())
    for canary_index, width in enumerate(int(v) for v in args.canary_lengths.split(',') if v):
        row = db.execute("SELECT id FROM examples WHERE split='train' AND token_count>=? ORDER BY token_count,id LIMIT 1", (width,)).fetchone()
        if row is None:
            raise ValueError('canary exceeds data')
        example = load_example(db, row[0])
        example['token_ids'] = example['token_ids'][:width]
        torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        metrics = response(example, backward=True)
        if args.mode == 'dp':
            sum_replica_gradients(compressor.parameters())
        old._normalize_gradients(compressor.parameters(), metrics['tokens']*(world if args.mode == 'dp' else 1))
        norm = float(compressor.clip_grad_norm_(1.0))
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError('canary gradient invalid')
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        canary = dict(passed=True, optimizer_updated=False, **metrics, grad_norm=norm,
            seconds=time.monotonic()-began, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        if rank == 0:
            old._atomic_json(output / f'canary-{width}-{canary_index}.json', canary)
            print(json.dumps(dict(phase='canary_passed', **canary)), flush=True)
        torch.cuda.empty_cache()
    if args.canary_lengths:
        verify_unchanged_weights('preflight-fixed-weight-verification.json')
        torch.set_rng_state(canary_rng[0]); torch.cuda.set_rng_state(canary_rng[1]); random.setstate(canary_rng[2])

    writer = SummaryWriter(str(output/'tensorboard'), purge_step=step+1) if rank == 0 and not args.benchmark else None
    def validate():
        cpu, cuda, rng = torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()
        compressor.eval()
        began = time.monotonic()
        ids = [x[0] for x in db.execute("SELECT id FROM examples WHERE split='validation' ORDER BY id")]
        assignments = assign_batch(ids, counts, 0, len(ids), world) if args.mode == 'dp' else [list(range(len(ids)))]*world
        totals = [0., 0., 0., 0.]
        length_totals = {n:[0.,0.,0.,0.] for n in (8192,16384,32768,65536)}
        for pos in assignments[rank]:
            m = response(load_example(db, ids[pos]), backward=False)
            for i, x in enumerate([m['reverse_kl']*m['tokens'],m['top1_agreement']*m['tokens'],m['tokens'],1]):
                totals[i] += x
                length_totals[bucket(counts[ids[pos]])][i] += x
        per_length = torch.tensor(list(length_totals.values()), device=device, dtype=torch.float64)
        dist.all_reduce(per_length)
        if args.mode == 'tp':
            per_length /= world
        by_length = {}
        for length, sums in zip(length_totals, per_length.tolist()):
            if sums[3]:
                by_length[str(length)] = dict(reverse_kl=sums[0]/sums[2],top1_agreement=sums[1]/sums[2],tokens=int(sums[2]),documents=int(sums[3]))
        values = torch.tensor(totals, device=device, dtype=torch.float64)
        dist.all_reduce(values)
        if args.mode == 'tp':
            values /= world
        if int(values[3]) != len(ids):
            raise ValueError('validation coverage mismatch')
        row = dict(global_step=step, split='validation', optimizer_updated=False,
                   reverse_kl=float(values[0]/values[2]), top1_agreement=float(values[1]/values[2]),
                   tokens=int(values[2]), documents=int(values[3]), seconds=time.monotonic()-began, by_length=by_length)
        if rank == 0:
            with (output/'validation.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            for k in ('reverse_kl','top1_agreement'):
                if writer:
                    writer.add_scalar('validation/'+k, row[k], step)
            if writer:
                for length, metrics in by_length.items():
                    for key in ('reverse_kl','top1_agreement'):
                        writer.add_scalar(f'validation_{length}/{key}',metrics[key],step)
                writer.flush()
            print(json.dumps(dict(phase='validation_complete', **row)), flush=True)
        compressor.train()
        torch.set_rng_state(cpu); torch.cuda.set_rng_state(cuda); random.setstate(rng)
        torch.cuda.empty_cache()
    if args.initial_validation:
        validate()
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGUSR1, request_stop)
    initial_step, initial_cursor = step, cursor
    began, last_saved = time.monotonic(), -1
    stop = torch.tensor(0, device=device)
    grid = ([tuple(map(int,x.split(':'))) for x in args.benchmark_grid.split(',')]
            if args.benchmark_grid else [(projection,threshold)])
    if args.benchmark_grid and not args.benchmark:
        raise ValueError('benchmark grid cannot update optimizer')
    benchmark_rows = []
    configurations = grid if args.benchmark else [(projection,threshold)]
    for configuration_index, (projection,threshold) in enumerate(configurations):
        epochs = range(args.benchmark_passes) if args.benchmark else range(initial_cursor.epoch,source['epochs'])
        for epoch in epochs:
            data_epoch = initial_cursor.epoch if args.benchmark else epoch
            order = training_order(db, source['seed'], data_epoch)
            cumulative_source = [0]
            for key in order:
                cumulative_source.append(cumulative_source[-1] + counts[key])
            first = initial_cursor.event_index if data_epoch == initial_cursor.epoch else 0
            if initial_cursor.shard_index or initial_cursor.chunk_index:
                raise ValueError('unexpected cursor dimensions')
            final = min(len(order), first+args.benchmark_batches*opt['grad_accumulation']) if args.benchmark else len(order)
            for start in range(first, final, opt['grad_accumulation']):
                end = min(start+opt['grad_accumulation'],final)
                positions = assign_batch(order,counts,start,end,world) if args.mode == 'dp' else [list(range(start,end))]*world
                token_count = sum(counts[order[i]]+1 for i in range(start,end))
                optimizer.zero_grad(set_to_none=True)
                dist.barrier(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                local = [0.,0.,0.,0.]
                work_started = time.monotonic()
                for pos in positions[rank]:
                    t = time.monotonic()
                    m = response(load_example(db,order[pos]),backward=True)
                    for i,x in enumerate([m['reverse_kl']*m['tokens'],m['top1_agreement']*m['tokens'],m['tokens'],1]):
                        local[i] += x
                    with (output/f'events-rank{rank}.jsonl').open('a') as f:
                        f.write(json.dumps(dict(epoch=data_epoch,index=pos,id=order[pos],seconds=time.monotonic()-t,**m))+'\n')
                torch.cuda.synchronize()
                work_seconds = time.monotonic()-work_started
                if args.mode == 'dp':
                    sum_replica_gradients(compressor.parameters())
                old._normalize_gradients(compressor.parameters(), token_count)
                norm = float(compressor.clip_grad_norm_(1.0))
                if not math.isfinite(norm) or norm <= 0:
                    raise FloatingPointError('invalid gradient')
                if any(p.requires_grad or p.grad is not None for p in target.model.parameters()):
                    raise RuntimeError('frozen reader received parameter gradients')
                lr = learning_rate(step,peak=opt['learning_rate'],warmup=opt['warmup_steps'],
                    total=opt['total_steps'],minimum=opt['minimum_learning_rate'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                if not args.benchmark:
                    optimizer.step(); step += 1
                    if not old._parameters_finite(compressor):
                        raise FloatingPointError('nonfinite compressor weights')
                values = torch.tensor(local,device=device,dtype=torch.float64)
                dist.all_reduce(values)
                if args.mode == 'tp':
                    values /= world
                if int(values[2]) != token_count or int(values[3]) != end-start:
                    raise ValueError('global document/token coverage mismatch')
                torch.cuda.synchronize()
                stats = dict(rank=rank,work_seconds=work_seconds,batch_seconds=time.monotonic()-before,
                             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                rank_stats = [None]*world
                dist.all_gather_object(rank_stats,stats)
                cursor = old.QueueCursor(data_epoch+1,0,0,0) if end == len(order) else old.QueueCursor(data_epoch,0,end,0)
                row = dict(global_step=step, epoch=data_epoch, start_position=start,end_position=end,
                    source_tokens_seen=cumulative_source[end],epoch_progress=end/len(order),
                    responses=end-start,tokens=token_count,reverse_kl=float(values[0]/token_count),
                    loss=float(values[0]/token_count),top1_agreement=float(values[1]/token_count),
                    grad_norm=norm,learning_rate=lr,elapsed_seconds=time.monotonic()-began,
                    batch_seconds=max(s['batch_seconds'] for s in rank_stats),
                    gpu_peak_allocated_mib=max(s['peak_allocated_mib'] for s in rank_stats),
                    assignments=positions,rank_stats=rank_stats,target_trainable_parameters=0,
                    cursor=dataclasses.asdict(cursor),configuration_index=configuration_index,
                    projection_tokens=projection,reader_checkpoint_after_tokens=threshold,
                    benchmark_pass=epoch if args.benchmark else None,optimizer_updated=not args.benchmark)
                if rank == 0:
                    with (output/('benchmark.jsonl' if args.benchmark else 'metrics.jsonl')).open('a') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                    if writer:
                        writer.add_scalar('train/source_tokens_seen',cumulative_source[end],step)
                        writer.add_scalar('train/epoch_progress',end/len(order),step)
                        for k in ('loss','reverse_kl','top1_agreement','grad_norm','learning_rate','batch_seconds','gpu_peak_allocated_mib'):
                            writer.add_scalar('train/'+k,row[k],step)
                        writer.add_scalar('train/tokens_per_second',token_count/row['batch_seconds'],step)
                        writer.flush()
                    print(json.dumps(row,allow_nan=False),flush=True)
                optimizer.zero_grad(set_to_none=True)
                if args.benchmark:
                    benchmark_rows.append(row)
                    continue
                stop = torch.tensor(int(stopping or (output/'STOP_AFTER_RESPONSE').exists()
                    or (args.max_additional_steps and step-initial_step>=args.max_additional_steps)),device=device)
                dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if bool(stop) or end == len(order) or (args.save_interval and step%args.save_interval==0):
                    save_full_checkpoint(output,compressor,optimizer,mode=args.mode,rank=rank,step=step,
                        cursor=cursor,contract_hash=contract_hash)
                    last_saved = step
                    old._prune_checkpoints(output,keep_last=args.keep_last_checkpoints,rank=rank)
                if (args.validation_interval and step%args.validation_interval==0) or end == len(order):
                    validate()
                if bool(stop):
                    break
            if not args.benchmark and bool(stop):
                break
    if args.benchmark:
        verify_unchanged_weights('fixed-weight-verification.json')
    if rank == 0:
        if args.benchmark:
            old._atomic_json(output/'benchmark-result.json',dict(execution=execution,optimizer_updated=False,rows=benchmark_rows))
        else:
            old._atomic_json(output/'completion.json',dict(global_step=step,cursor=dataclasses.asdict(cursor),
                checkpoint=str(output/f'checkpoint-{last_saved:08d}'),all_epochs_complete=cursor.epoch>=source['epochs'],
                stopped_on_request=stopping or (output/'STOP_AFTER_RESPONSE').exists()))
        if writer:
            writer.close()
    db.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
