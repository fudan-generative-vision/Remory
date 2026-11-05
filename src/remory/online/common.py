"""Checkpoint identities and durable state; safe to import without PyTorch."""
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from ..types import digest

ENCODER_VERSION = ContextVar('remory_encoder_version', default=None)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            temporary.chmod(0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('a') as stream:
        os.chmod(path, 0o600)
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def version_id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid REMORY checkpoint ID')
    return value


def checkpoint(path, *, verify=False):
    path = Path(path).resolve()
    config, manifest = read(path / 'config.json'), read(path / 'manifest.json')
    if not config or not manifest or not (path / 'model.safetensors').is_file():
        raise ValueError('Incomplete serving checkpoint: ' + str(path))
    if manifest.get('config_sha256') != digest(config):
        raise ValueError('Checkpoint configuration checksum differs')
    ident = version_id(manifest.get('safetensors_sha256'))
    if verify and checksum(path / 'model.safetensors') != ident:
        raise ValueError('Checkpoint weight checksum differs')
    receipt = read(path / 'session-training.json', {})
    return dict(id=ident, path=str(path), compatible=digest(config),
        label=receipt.get('label', 'Released checkpoint'), step=receipt.get('step'),
        created=receipt.get('created', (path / 'manifest.json').stat().st_mtime),
        validation=receipt.get('validation'))


def resolve_version(root, base, ident, *, verify=False):
    version_id(ident)
    initial = checkpoint(base)
    if ident == initial['id']:
        return checkpoint(base, verify=verify)
    root = Path(root).resolve()
    item = read(root / 'versions' / (ident + '.json'))
    if not item:
        raise ValueError('Unknown REMORY checkpoint; select it with /remory load')
    path = Path(item['path']).resolve()
    if not path.is_relative_to(root / 'checkpoints') or path.parent.name.startswith('.'):
        raise ValueError('Checkpoint is outside the online training directory')
    result = checkpoint(path, verify=verify)
    if result['id'] != ident or result['compatible'] != initial['compatible']:
        raise ValueError('Incompatible REMORY checkpoint')
    return result
