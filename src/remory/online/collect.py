"""Capture exact pre-compaction inputs; train only after client acknowledgement."""
import logging
from pathlib import Path
import re
import time
import uuid

import numpy as np

from ..types import Prepared, digest, token_ids
from .common import checkpoint, checksum, locked, read, write_json


class Pool:
    def __init__(self, root, *, config, base, actor, identity, serving_gpus):
        self.root = Path(root).expanduser().resolve()
        self.base = Path(base).resolve()
        self.config = config
        meta = dict(schema='remory.online.v1', root=str(self.root), checkpoint=str(self.base),
            actor=str(Path(actor).resolve()), config=config, identity=identity,
            base_encoder=checkpoint(base)['id'], serving_gpus=list(serving_gpus))
        with locked(self.root / 'pool.lock'):
            old = read(self.root / 'pool.json')
            if old and (old['identity'] != identity or old['base_encoder'] != meta['base_encoder']):
                raise ValueError('Online training directory belongs to a different model/checkpoint')
            write_json(self.root / 'pool.json', meta)
        self.meta = meta

    def capture(self, engine, owner, args, memory, opening_ids):
        """Called only after successful encoding; no sample from folded children."""
        if args.get('branches'):
            return None
        prefix = token_ids(args['prefix_ids'], 'prefix_ids')
        history = token_ids(args['history_ids'], 'history_ids', empty=True)
        opening = token_ids(opening_ids, 'capture_opening_ids')
        if len(opening) > 256:
            raise ValueError('Assistant opening exceeds 256 tokens')
        previous = args.get('previous')
        if previous:
            source = engine._prepare(owner=owner, handle=previous, continuation_ids=history)
        else:
            source = Prepared(prefix + history)
        key = uuid.uuid4().hex
        folder = self.root / 'samples' / key
        folder.mkdir(parents=True, mode=0o700)
        memory_file = None
        if source.memory is not None:
            memory_file = 'memory.npy'
            np.save(folder / memory_file, source.memory, allow_pickle=False)
            (folder / memory_file).chmod(0o600)
        sample = dict(schema='remory.online.frontier.v1', id=key, created=time.time(),
            owner_sha256=digest(owner), handle=memory.handle, prefix_ids=list(prefix),
            source_ids=list(source.input_ids[len(prefix):]), opening_ids=list(opening),
            memory_positions=list(source.memory_positions), memory_depths=list(source.slot_depths),
            memory_file=memory_file, memory_sha256=checksum(folder / memory_file) if memory_file else None,
            context_limit=engine.backend.context_limit, deployment_sha256=digest(self.config))
        sample['frontier_sha256'] = digest(sample)
        write_json(folder / 'frontier.json', sample)
        write_json(folder / 'summary.json', dict(ids=list(memory.summary_ids), identity=memory.handle,
            summary_sha256=digest(list(memory.summary_ids))))
        return key

    def accept(self, ident, owner):
        if not re.fullmatch('[0-9a-f]{32}', ident):
            raise ValueError('Invalid sample ID')
        folder = self.root / 'samples' / ident
        sample = read(folder / 'frontier.json')
        if not sample or sample['owner_sha256'] != digest(owner):
            raise KeyError('Unknown sample for this session')
        write_json(folder / 'accepted.json', dict(identity=sample['handle'], accepted_at=time.time()))
        return {'accepted': True}


def valid_sample(folder):
    folder = Path(folder)
    x, summary, accepted = (read(folder / name) for name in
        ('frontier.json', 'summary.json', 'accepted.json'))
    if not x or not summary or not accepted or accepted['identity'] != summary['identity']:
        raise ValueError('Sample has no matching installed summary')
    if x['schema'] != 'remory.online.frontier.v1' or x['handle'] != summary['identity']:
        raise ValueError('Unsupported or mismatched frontier')
    original = dict(x)
    claimed = original.pop('frontier_sha256')
    if digest(original) != claimed or summary['summary_sha256'] != digest(summary['ids']):
        raise ValueError('Captured frontier or summary changed')
    for name in ('prefix_ids', 'source_ids', 'opening_ids'):
        token_ids(x[name], name)
    token_ids(summary['ids'], 'summary_ids')
    positions, depths = x['memory_positions'], x['memory_depths']
    if (positions != sorted(set(positions)) or len(positions) != len(depths)
        or any(type(p) is not int or not len(x['prefix_ids']) <= p < len(x['prefix_ids'] + x['source_ids']) for p in positions)
        or any(type(d) is not int or d < 1 for d in depths)):
        raise ValueError('Invalid source memory coordinates')
    if x['memory_file']:
        if x['memory_file'] != 'memory.npy' or checksum(folder / 'memory.npy') != x['memory_sha256']:
            raise ValueError('Captured memory tensor changed')
        memory = np.load(folder / 'memory.npy', allow_pickle=False)
        if memory.ndim != 2 or len(memory) != len(positions) or not np.isfinite(memory).all():
            raise ValueError('Invalid source memory tensor')
    elif positions:
        raise ValueError('Missing source memory tensor')
    return x, summary


def capture_safely(pool, *args):
    try:
        return pool.capture(*args)
    except Exception:
        logging.getLogger(__name__).exception('Online sample capture failed; compaction remains valid')
        return None
