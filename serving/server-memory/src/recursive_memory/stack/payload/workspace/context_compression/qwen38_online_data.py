"""Immutable causal samples and a resumable, host-side online replay queue."""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_sample(x):
    if x['dataset'] not in {'tb4', 'jobbench'}:
        raise ValueError('unexpected training dataset')
    for key in ['prefix_ids', 'raw_history_ids', 'source_ids', 'opening_ids', 'target_ids', 'summary_ids']:
        if not isinstance(x[key], list) or not x[key] or any(type(t) is not int or t < 0 for t in x[key]):
            raise ValueError(f'invalid {key}')
    n = len(x['prefix_ids']) + len(x['raw_history_ids']) + len(x['opening_ids']) + len(x['target_ids'])
    if n > 262144:
        raise ValueError('teacher context exceeds 262144; truncation is forbidden')
    if x.get('teacher_finish_type') != 'stop' or x.get('summary_accepted') is not True:
        raise ValueError('unfinished teacher or unaccepted summary')
    pos, depths = x.get('source_memory_positions', []), x.get('source_memory_depths', [])
    if pos != sorted(set(pos)) or len(pos) != len(depths):
        raise ValueError('invalid recursive memory coordinates')
    if any(type(p) is not int or not len(x['prefix_ids']) <= p < len(x['prefix_ids']) + len(x['source_ids']) for p in pos):
        raise ValueError('memory escapes causal source')
    if any(type(d) is not int or d <= 0 for d in depths):
        raise ValueError('invalid logical recursion depth')
    if bool(pos) != bool(x.get('source_memory_file')):
        raise ValueError('missing recursive tensor')
    if x.get('has_images'):
        raise ValueError('text reader cannot silently discard image evidence')
    if x.get('branch_protocol'):
        from context_compression.qwen38_online_protocol import PROTOCOL
        check = x.get('summary_validation', {})
        if (x['branch_protocol'] != PROTOCOL or check.get('protocol') != PROTOCOL
                or check.get('format_valid') is not True or check.get('natural_completion') is not True
                or check.get('summary_ids_sha256') != digest(x['summary_ids'])):
            raise ValueError('missing or stale strict summary validation')
        if x.get('branch_parent_kind') != 'actual_rollout_frontier':
            raise ValueError('branch is not rooted in the real rollout')
        frontier = x.get('rollout_frontier')
        if (x.get('source_memory_file') != (frontier['memory_file'] if frontier else None)
                or x.get('source_memory_depths', []) != (frontier['memory_depths'] if frontier else [])):
            raise ValueError('branch memory differs from the installed frontier')
    return x


class ReplayQueue:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'queue.sqlite', timeout=60)
        # DELETE journaling + explicit transactions works on this shared filesystem.
        self.db.execute('PRAGMA busy_timeout=60000')
        self.db.execute('CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, dataset TEXT, path TEXT, tokens INTEGER, consumed INTEGER DEFAULT 0)')
        self.db.commit()

    def publish(self, x):
        validate_sample(x)
        key = digest({k: x[k] for k in ['dataset', 'task_id', 'rollout_id', 'round', 'policy_version']})
        path = self.root / 'samples' / (key + '.json')
        if path.exists():
            if json.loads(path.read_text()) != x:
                raise ValueError('immutable sample identity collision')
        else:
            atomic_json(path, x)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO samples(id,dataset,path,tokens) VALUES(?,?,?,?)', (key, x['dataset'], str(path), len(x['target_ids'])))
        return key

    def batch(self, consumed_ids, threshold=16384, prefer='tb4', allow_partial=False, min_samples=1, max_wait_seconds=None):
        # The checkpoint, not this mutable database, is the consumption authority.
        got = []
        for domain in [prefer, 'jobbench' if prefer == 'tb4' else 'tb4']:
            for key, path, tokens in self.db.execute('SELECT id,path,tokens FROM samples WHERE dataset=? ORDER BY rowid', (domain,)):
                if key in consumed_ids:
                    continue
                got.append((key, path, tokens))
                if sum(x[2] for x in got) >= threshold and len(got)>=min_samples:
                    return got
        expired=bool(got) and max_wait_seconds is not None and time.time()-min(Path(x[1]).stat().st_mtime for x in got)>=max_wait_seconds
        return got if (allow_partial or expired) and len(got)>=min_samples else []
