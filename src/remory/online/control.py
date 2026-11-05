"""Session controls for a shared, opt-in training pool."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .common import checkpoint, locked, read, write_json
from .devices import inventory, resolve, validate_devices


def command(params):
    items = params.get('input')
    if (not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict)
            or items[0].get('type') != 'text'):
        return None
    text = items[0].get('text', '').strip()
    if not re.match(r'^[\\/]remory(?:\s|$)', text):
        return None
    return text.split()[1:] or ['help']


def alive(owner):
    try:
        fields = Path(f"/proc/{int(owner['pid'])}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == owner['start']
    except (OSError, ValueError, KeyError):
        return False


class Control:
    def __init__(self, session, info, url):
        meta = info.get('online_training')
        if not meta or meta.get('schema') != 'remory.online.v1':
            raise ValueError('Restart SGLang with --online-training-dir first')
        self.session = Path(session).resolve()
        self.root = self.session / 'remory'
        self.pool = Path(meta['root']).resolve()
        self.base = Path(meta['checkpoint'])
        if read(self.pool / 'pool.json') != meta:
            raise ValueError('Online training needs the same local/shared filesystem as the server')
        self.info, self.url, self.meta = info, url, meta

    def configure(self, *, gpus=None, python=None):
        with locked(self.pool / 'control.lock'):
            cfg = read(self.pool / 'config.json')
            previous = dict(cfg) if cfg is not None else None
            if cfg is None:
                if not gpus:
                    raise ValueError('Choose reserved GPUs with --training-gpus (for example 2,3)')
                cfg = dict(schema='remory.online.config.v1', learning_rate=1e-6, max_steps=100,
                    save_every=10, teacher_max_tokens=16384, minimum_samples=2,
                    min_free_mib=100000, max_used_mib=1024,
                    python=sys.executable)
            if gpus is not None:
                cfg['gpus'] = resolve([x.strip() for x in gpus.split(',')], inventory())
            if python:
                cfg['python'] = str(Path(python).expanduser().absolute())
            cfg.update(url=self.url, checkpoint=str(self.base), target=self.meta['actor'],
                       source=str(Path(__file__).resolve().parents[3]))
            validate_devices(cfg['gpus'], self.meta['serving_gpus'])
            if not (Path(cfg['source']) / 'training/context_compression/qwen38_online_objective.py').is_file():
                raise ValueError('Online training requires an editable install from the repository clone')
            if cfg == previous:
                return
            if alive(read(self.pool / 'worker.json', {})):
                raise ValueError('Stop the shared trainer before changing its configuration')
            write_json(self.pool / 'config.json', cfg)

    def selection(self):
        return read(self.root / 'pending-load.json') or read(self.root / 'active.json') or checkpoint(self.base)

    def candidates(self):
        base = checkpoint(self.base)
        active = read(self.root / 'active.json', base)['id']
        result, seen = [], set()
        for path in [self.base, *sorted((self.pool / 'checkpoints').glob('*/serving'))]:
            if path != self.base and (path.parent.name.startswith('.')
                    or not path.resolve().is_relative_to(self.pool / 'checkpoints')):
                continue
            try:
                item = checkpoint(path)
                if item['compatible'] != base['compatible'] or item['id'] in seen:
                    continue
                item['current'] = item['id'] == active
                result.append(item)
                seen.add(item['id'])
            except (OSError, ValueError, KeyError):
                continue
        return sorted(result, key=lambda x: x['created'], reverse=True)

    def select(self, ident):
        options = {item['id']: item for item in self.candidates()}
        if ident not in options:
            raise ValueError('Checkpoint is no longer available; open the picker again')
        selected = checkpoint(options[ident]['path'], verify=True)
        with locked(self.root / 'control.lock'):
            active = read(self.root / 'active.json', checkpoint(self.base))
            if active['id'] == ident:
                (self.root / 'pending-load.json').unlink(missing_ok=True)
                return 'REMORY: this checkpoint is already active.'
            write_json(self.pool / 'versions' / (ident + '.json'), selected)
            write_json(self.root / 'pending-load.json', selected)
        return ('REMORY: selected ' + selected['label'] + '. It will apply at the next successful '
                'compaction; existing memory and other sessions keep their versions.')

    def applied(self, ident):
        with locked(self.root / 'control.lock'):
            item = read(self.pool / 'versions' / (ident + '.json'))
            if ident == self.meta['base_encoder']:
                item = checkpoint(self.base)
            if item is None:
                raise ValueError('Applied checkpoint is missing from the local registry')
            write_json(self.root / 'active.json', item)
            if read(self.root / 'pending-load.json', {}).get('id') == ident:
                (self.root / 'pending-load.json').unlink(missing_ok=True)

    def train(self):
        with locked(self.pool / 'control.lock'):
            cfg = read(self.pool / 'config.json')
            if not cfg:
                raise ValueError('Configure --training-gpus and --train-python when launching r-codex')
            validate_devices(cfg['gpus'], self.meta['serving_gpus'])
            (self.pool / 'STOP').unlink(missing_ok=True)
            if alive(read(self.pool / 'worker.json', {})):
                return 'REMORY: trainer already running.\n' + self.status()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
            source = Path(cfg['source'])
            env['PYTHONPATH'] = os.pathsep.join([str(source / 'src'), str(source / 'training')])
            env.pop('PYTHONHOME', None)
            with (self.pool / 'worker.log').open('a') as log:
                os.chmod(log.name, 0o600)
                child = subprocess.Popen([cfg['python'], '-m', 'remory.online.worker', str(self.pool)],
                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            write_json(self.pool / 'worker.json', dict(pid=child.pid,
                start=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        return ('REMORY: background trainer started. It waits for complete samples and the configured '
                'GPUs. Use /remory status for progress, /remory load for checkpoints.')

    def status(self):
        state = read(self.pool / 'status.json', {})
        active = read(self.root / 'active.json', checkpoint(self.base))
        pending = read(self.root / 'pending-load.json')
        phase = state.get('phase', 'not started')
        if phase not in {'not started', 'stopped', 'failed'} and not alive(read(self.pool / 'worker.json', {})):
            phase = 'interrupted (run /remory train to resume)'
        return (f"REMORY: {phase}; step {state.get('step', 0)}; samples {state.get('samples', 0)}; "
                f"active {active['label']}" + (f"; pending {pending['label']}" if pending else '') +
                (f"\n{state['detail']}" if state.get('detail') else ''))

    def stop(self):
        write_json(self.pool / 'STOP', dict(requested_at=time.time()))
        return ('REMORY: stopping the shared trainer after its current step and saving a checkpoint. '
                'Opted-in sessions continue collecting samples.')

    def execute(self, args):
        if len(args) == 1 and args[0] in {'train', 'status', 'stop'}:
            return getattr(self, args[0])()
        return 'REMORY commands: /remory train | /remory load | /remory status | /remory stop'
