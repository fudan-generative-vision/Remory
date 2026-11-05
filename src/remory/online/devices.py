"""Explicit training GPU selection, separate from the inference devices."""
from contextlib import ExitStack, contextmanager
import os
import subprocess

from .common import locked


def inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=15, check=True)
    rows = [tuple(part.strip() for part in line.split(',')) for line in result.stdout.splitlines()]
    return {index: dict(uuid=ident, used=int(used), free=int(free)) for index, ident, used, free in rows}


def resolve(devices, available):
    resolved = []
    for device in devices:
        matches = [v['uuid'] for k, v in available.items() if device in (k, v['uuid'])]
        if len(matches) != 1:
            raise ValueError('Unknown GPU index/UUID: ' + device)
        resolved += matches
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError('Select distinct training GPUs')
    return resolved


def serving_devices(start, count):
    available = inventory()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible is not None else list(available)
    devices = [x.strip() for x in devices][start:start + count]
    if len(devices) != count:
        raise ValueError('Cannot determine the inference GPUs for online training')
    return resolve(devices, available)


def validate_devices(devices, serving):
    if not isinstance(devices, list) or not devices or set(devices) & set(serving):
        raise ValueError('Training GPUs must be explicitly selected and exclude inference GPUs')
    if len(set(devices)) != len(devices):
        raise ValueError('Training GPUs must be distinct')


@contextmanager
def acquire(root, cfg, serving):
    devices = cfg['gpus']
    validate_devices(devices, serving)
    available = {v['uuid']: v for v in inventory().values()}
    if any(d not in available or available[d]['used'] > cfg['max_used_mib']
           or available[d]['free'] < cfg['min_free_mib'] for d in devices):
        yield False
        return
    with ExitStack() as stack:
        try:
            for ident in sorted(devices):
                stack.enter_context(locked(root / 'locks' / (ident + '.lock'), blocking=False))
        except BlockingIOError:
            yield False
            return
        yield True
