"""Frozen-reader distillation from accepted interactive compact frontiers."""
import fcntl
import os
from pathlib import Path
import random
import signal
import sys
import time
import uuid
from datetime import timedelta

import torch
import numpy as np
from safetensors.torch import load_file, save_file
import torch.distributed as dist

from .common import read, digest, write_json, checksum, checkpoint
from .collect import valid_sample
from .distributed import reduce_gradients
from ..models.load import load_compressor, resolve_checkpoint
from context_compression.qwen38_online_objective import OnlineObjective, OnlineEvent
from context_compression.qwen38_multimodal_target import FrozenQwen38MultimodalTarget, load_qwen38_multimodal_model


class FrontierObjective(OnlineObjective):
    def raw_teacher_and_source(self, event):
        # Teacher sees the exact pre-compact frontier, including earlier residuals.
        # Its continuation is appended only here, never to the compressor source.
        ids = event.prefix_ids + event.history_ids + event.opening_ids
        with self._frozen_eval():
            inputs = self._embed(ids + event.target_ids).clone()
            if event.source_memory is not None:
                inputs[:, list(event.source_memory_positions)] = event.source_memory.to(inputs)
            features, hidden = self._capture(inputs, len(event.prefix_ids), len(event.prefix_ids + event.history_ids))
            teacher = hidden[:, len(ids)-1:len(ids)-1+len(event.target_ids)].detach().to('cpu', copy=True)
        return features, teacher


def load_event(folder):
    x, summary = valid_sample(folder)
    teacher = read(folder / 'teacher.json')
    if (teacher['finish'] != 'stop' or teacher['frontier_sha256'] != x['frontier_sha256']
            or teacher['summary_sha256'] != summary['summary_sha256']):
        raise ValueError('Teacher is not bound to this accepted compact')
    memory = torch.from_numpy(np.load(folder / x['memory_file'], allow_pickle=False)) if x['memory_file'] else None
    event = OnlineEvent(key=x['id'], turn=0, prefix_ids=tuple(x['prefix_ids']),
        history_ids=tuple(x['source_ids']), opening_ids=tuple(x['opening_ids']), target_ids=tuple(teacher['ids']),
        source_memory=memory, source_memory_positions=tuple(x['memory_positions']), source_memory_depths=tuple(x['memory_depths']))
    return event, summary


def main(run):
    run = Path(run)
    cfg = read(run / 'config.json')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != cfg['world_size'] or torch.cuda.device_count() != world:
        raise ValueError('Training requires exactly the configured reserved GPUs')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=2))
    lease = None
    if rank == 0:
        lease = (run / 'learner.lock').open('a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(cfg['root'])
    initial = Path(cfg['checkpoint'])
    _, deployment = resolve_checkpoint(str(initial))
    torch.manual_seed(20261015)
    random.seed(20261015)
    target = FrozenQwen38MultimodalTarget(load_qwen38_multimodal_model(cfg['target'], dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': device}),
        target_layer_ids=deployment['compressor']['target_layer_ids'], gradient_checkpointing=True)
    model, _ = load_compressor(str(initial), device=device, dtype=torch.float32)
    model.requires_grad_(True).train()
    model.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    weighting = deployment.get('position_weighting', {})
    objective = FrontierObjective(target, model, deployment['summary_contract'],
        memory_budget=deployment['residual']['memory_budget'], feature_residency='cpu',
        node_batch_size=2, projection_tokens=128, offload_after_tokens=16384,
        reader_checkpoint_after_tokens=16384,
        **{key: weighting[key] for key in ('alpha', 'tau', 'power') if key in weighting})
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(root / 'tensorboard' / run.name)) if rank == 0 else None
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGUSR1, stopping)
    signal.signal(signal.SIGINT, stopping)
    consumed = set(read(root / 'consumed.json', []))
    step = last_saved = cfg.get('initial_step', 0)
    saved = read(run / 'latest.json')
    resume_dir = Path(saved['path']) if saved else initial.parent
    if (resume_dir / 'training.pt').is_file():
        state = torch.load(resume_dir / 'training.pt', weights_only=True, map_location='cpu')
        if saved:
            restored = checkpoint(resume_dir / 'serving', verify=True)
            if restored['compatible'] != digest(deployment):
                raise ValueError('Training checkpoint changed its deployment contract')
            model.load_state_dict(load_file(str(resume_dir / 'serving/model.safetensors')))
        optimizer.load_state_dict(state['optimizer'])
        rngs = state.get('rngs')
        if rngs is None:
            rngs = [dict(cpu=state['cpu_rng'], cuda=state['cuda_rng'], python=random.getstate())]
        if len(rngs) not in (1, world):
            raise ValueError('Checkpoint has incompatible training ranks')
        rng = rngs[rank % len(rngs)]
        torch.set_rng_state(rng['cpu'])
        torch.cuda.set_rng_state(rng['cuda'], device=device)
        random.setstate(rng['python'])
        consumed.update(state['consumed'])
        step = last_saved = state['step']
        del state
        if saved and rank == 0:
            # Repair publication pointers if interruption occurred between their
            # atomic writes. The optimizer checkpoint owns the consumed cursor.
            write_json(root / 'latest.json', dict(path=str(resume_dir),
                serving=str(resume_dir / 'serving'), step=step))
            write_json(root / 'consumed.json', sorted(consumed))

    def status(**fields):
        if rank == 0:
            write_json(root / 'status.json', dict(step=step, samples=len(cfg['samples']),
                gpus=cfg['gpus'], world_size=world, updated_at=time.time(), **fields))

    def save():
        nonlocal last_saved
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device), python=random.getstate())
        rngs = [None] * world
        dist.all_gather_object(rngs, rng)
        if rank != 0:
            dist.barrier()
            last_saved = step
            return
        status(phase='saving', detail='Saving REMORY weights, optimizer, sample cursor and per-rank RNG states.')
        parent = root / 'checkpoints'
        parent.mkdir(exist_ok=True, mode=0o700)
        dest = parent / f'{run.name}-step-{step:06d}-{uuid.uuid4().hex[:8]}'
        stage = parent / ('.' + uuid.uuid4().hex)
        serving = stage / 'serving'
        serving.mkdir(parents=True, mode=0o700)
        weights = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
        if not all(bool(torch.isfinite(v).all()) for v in weights.values()):
            raise RuntimeError('Refusing to publish nonfinite weights')
        save_file(weights, str(serving / 'model.safetensors'))
        write_json(serving / 'config.json', deployment)
        manifest = dict(read(initial / 'manifest.json'), world_size=world,
                        safetensors_sha256=checksum(serving / 'model.safetensors'),
                        config_sha256=digest(deployment), model_size_bytes=(serving / 'model.safetensors').stat().st_size)
        write_json(serving / 'manifest.json', manifest)
        write_json(serving / 'session-training.json', dict(label=f'Pooled step {step} ({run.name[-6:]})',
            step=step, created=time.time(), source=str(initial), world_size=world,
            validation='finite_weights_and_gradients'))
        torch.save(dict(optimizer=optimizer.state_dict(), rngs=rngs,
                       consumed=sorted(consumed), step=step), stage / 'training.pt')
        for p in stage.rglob('*'):
            if p.is_file():
                p.chmod(0o600)
                with p.open('rb') as f:
                    os.fsync(f.fileno())
        stage.rename(dest)
        write_json(run / 'latest.json', dict(path=str(dest), step=step))
        write_json(root / 'latest.json', dict(path=str(dest), serving=str(dest / 'serving'), step=step))
        write_json(root / 'consumed.json', sorted(consumed))
        last_saved = step
        dist.barrier()

    pending = [Path(p) for p in cfg['samples'] if Path(p).name not in consumed]
    stopped = False
    for offset in range(0, len(pending), world):
        stopping_now = torch.tensor(int(stop or (root / 'STOP').exists()), device=device)
        dist.all_reduce(stopping_now, op=dist.ReduceOp.MAX)
        if stopping_now.item():
            stopped = True
            break
        batch = pending[offset:offset + world]
        status(phase='training', next_step=step+1,
            detail='Computing frozen-reader distillation on GPUs ' + ','.join(cfg['gpus']) + '.')
        optimizer.zero_grad(set_to_none=True)
        metrics = dict(tokens=0, loss_sum=0.)
        if rank < len(batch):
            event, summary = load_event(batch[rank])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                metrics = objective.backward_response(event, summary)
        totals = torch.tensor([metrics['tokens'], metrics['loss_sum']], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        tokens, loss_sum = totals.tolist()
        reduce_gradients(model.parameters(), tokens)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)) or norm <= 0:
            raise RuntimeError('REMORY gradient is missing or nonfinite')
        if any(p.requires_grad or p.grad is not None for p in target.model.parameters()):
            raise RuntimeError('Frozen LLM received a parameter gradient')
        optimizer.step()
        step += 1
        consumed.update(p.name for p in batch)
        status(phase='training', loss=loss_sum/tokens, grad_norm=float(norm),
               target_tokens=int(tokens), consumed=len(consumed))
        if writer:
            writer.add_scalar('loss', loss_sum/tokens, step)
            writer.add_scalar('grad_norm', float(norm), step)
            writer.add_scalar('target_tokens', tokens, step)
            writer.flush()
        if step % cfg['save_every'] == 0:
            save()
    if step > last_saved:
        save()
    if writer:
        writer.close()
    stopped = stopped or stop or (root / 'STOP').exists()
    if rank == 0 and not stopped:
        write_json(run / 'completed.json', dict(step=step, completed_at=time.time()))
    status(phase='stopped' if stopped else 'completed', detail='Use /remory load to choose a checkpoint.')
    dist.destroy_process_group()
    if lease:
        lease.close()


if __name__ == '__main__':
    main(sys.argv[1])
