"""Persistent coordinator: completed teachers, reserved GPUs, resumable updates."""
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..backends.sglang_http import SGLangHTTPBackend
from ..types import Prepared, digest
from .collect import valid_sample
from .common import locked, read, write_json
from .devices import acquire, validate_devices


def teacher(folder, cfg, meta):
    sample, summary = valid_sample(folder)
    ids = sample['prefix_ids'] + sample['source_ids'] + sample['opening_ids']
    budget = min(cfg['teacher_max_tokens'], sample['context_limit'] - len(ids))
    if budget < 1:
        write_json(folder / 'excluded.json', {'reason': 'No teacher output space; input is never truncated'})
        return
    source = Prepared(tuple(ids), tuple(sample['memory_positions']),
        np.load(folder / 'memory.npy', allow_pickle=False) if sample['memory_file'] else None,
        tuple(sample['memory_depths']))
    backend = SGLangHTTPBackend(cfg['url'], meta['config'], meta['actor'],
        context_limit=sample['context_limit'] + 1, timeout=1800)
    try:
        result = backend.generate(source, max_new_tokens=budget,
            sampling={'temperature': 0.6, 'top_p': 0.95, 'top_k': 20})
    finally:
        backend.close()
    kind = result.finish_reason
    if isinstance(kind, dict):
        kind = kind.get('type')
    if kind != 'stop' or not result.output_ids:
        write_json(folder / 'excluded.json', {'reason': 'Teacher did not complete: ' + str(kind)})
        return
    # A hypothetical next assistant step. Tool calls remain tokens and are never executed.
    write_json(folder / 'teacher.json', dict(ids=result.output_ids, finish='stop',
        frontier_sha256=sample['frontier_sha256'], summary_sha256=summary['summary_sha256'],
        input_tokens=len(ids), created=time.time()))


def validate_config(cfg, meta):
    validate_devices(cfg['gpus'], meta['serving_gpus'])
    for key in ('minimum_samples', 'max_steps', 'save_every', 'teacher_max_tokens', 'min_free_mib'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('Invalid online training setting: ' + key)
    if type(cfg['max_used_mib']) is not int or cfg['max_used_mib'] < 0:
        raise ValueError('Invalid max_used_mib')
    if not isinstance(cfg['learning_rate'], (float, int)) or not 0 < cfg['learning_rate'] < math.inf:
        raise ValueError('Invalid learning rate')


def main(directory):
    root = Path(directory).resolve()
    cfg, meta = read(root / 'config.json'), read(root / 'pool.json')
    validate_config(cfg, meta)
    with locked(root / 'worker.lock', blocking=False):
        while not (root / 'STOP').exists():
            folders = sorted(p.parent for p in (root / 'samples').glob('*/accepted.json'))
            latest_step = read(root / 'latest.json', {}).get('step', 0)
            def status(phase, detail, samples):
                write_json(root / 'status.json', dict(phase=phase, detail=detail,
                    step=latest_step, samples=samples, gpus=cfg['gpus'], updated_at=time.time()))
            for folder in folders:
                if (folder / 'teacher.json').exists() or (folder / 'excluded.json').exists():
                    continue
                status('teacher', 'Generating a teacher continuation; tools are not executed.', len(folders))
                try:
                    sample, _ = valid_sample(folder)
                    if sample['deployment_sha256'] != digest(meta['config']):
                        raise ValueError('Different reader or deployment contract')
                except (ValueError, KeyError, OSError) as exc:
                    write_json(folder / 'excluded.json', {'reason': str(exc)})
                    continue
                try:
                    teacher(folder, cfg, meta)
                except Exception as exc:
                    status('teacher_retry', str(exc), len(folders))
                    break
                if (root / 'STOP').exists():
                    break
            if (root / 'STOP').exists():
                break
            consumed = set(read(root / 'consumed.json', []))
            ready = [p for p in folders if (p / 'teacher.json').exists()
                and not (p / 'excluded.json').exists() and p.name not in consumed
                and read(p / 'frontier.json', {}).get('deployment_sha256') == digest(meta['config'])]
            previous = read(root / 'current-run.json', {})
            run = Path(previous['path']) if previous else None
            resume = run is not None and not (run / 'completed.json').exists()
            if len(ready) >= cfg['minimum_samples'] or resume:
                with acquire(root, cfg, meta['serving_gpus']) as available:
                    if available:
                        if not resume:
                            run = root / 'runs' / str(time.time_ns())
                            latest = read(root / 'latest.json', {})
                            write_json(run / 'config.json', dict(cfg, root=str(root),
                                checkpoint=latest.get('serving', cfg['checkpoint']),
                                samples=[str(p) for p in ready[:cfg['max_steps'] * len(cfg['gpus'])]],
                                world_size=len(cfg['gpus']), initial_step=latest.get('step', 0)))
                            write_json(root / 'current-run.json', {'path': str(run)})
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(cfg['gpus']),
                            PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
                        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                            env.pop(key, None)
                        env.setdefault('OMP_NUM_THREADS', '4')
                        env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
                        status('loading_reader', 'Loading frozen Qwen and trainable REMORY on reserved GPUs.', len(ready))
                        result = subprocess.run([cfg['python'], '-m', 'torch.distributed.run',
                            '--standalone', '--nnodes=1', '--nproc-per-node=' + str(len(cfg['gpus'])),
                            '--max-restarts=0', '--module', 'remory.online.learner', str(run)], env=env)
                        if result.returncode:
                            raise RuntimeError(f'Trainer exited with code {result.returncode}; see {root / "worker.log"}')
                        continue
                status('waiting_for_gpu', 'Waiting for the configured training GPUs to be unused.', len(ready))
            else:
                status('waiting_for_samples', f'Waiting for {cfg["minimum_samples"]} accepted compact samples.', len(ready))
            for _ in range(25):
                if (root / 'STOP').exists():
                    break
                time.sleep(.2)
        write_json(root / 'status.json', dict(phase='stopped',
            step=read(root / 'latest.json', {}).get('step', 0), updated_at=time.time()))


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except Exception as exc:
        write_json(Path(sys.argv[1]) / 'status.json', dict(phase='failed', detail=str(exc)))
        raise
