"""Private local transport for the native Codex TUI."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import queue
import stat
import threading
import time
import uuid

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import unix_connect


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def endpoint_path(session):
    # Unix socket names have a short kernel limit; the persistent path is too long.
    key = hashlib.sha256(str(Path(session).resolve()).encode()).hexdigest()[:16]
    directory = Path('/tmp') / f'remory-cli-{os.getuid()}-{key}'
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f'Interactive socket directory is not private: {directory}')
    return directory / 'app-server.sock'


def connect(path, child=None, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        if child is not None and child.poll() is not None:
            raise RuntimeError(f'Native app-server exited during startup: {child.returncode}')
        try:
            # Native 0.149 does not negotiate permessage-deflate on Unix sockets.
            return unix_connect(str(path), compression=None, max_size=None,
                                open_timeout=2, ping_timeout=None)
        except (OSError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


class RPC:
    def __init__(self, connection, directory):
        self.connection, self.directory = connection, Path(directory)
        self.next_id = 0
        self.responses = {}
        self.lock, self.send_lock = threading.Lock(), threading.Lock()
        self.latest = {}
        self.active_turn = None
        self.root_id = None
        self.closed = False
        self.pending = {}
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        try:
            with (self.directory / 'app-events.jsonl').open('a', buffering=1) as log:
                for raw in self.connection:
                    message = json.loads(raw)
                    log.write(json.dumps({'observed_at': time.time(), **message}, ensure_ascii=False) + '\n')
                    if 'id' in message and 'method' not in message:
                        with self.lock:
                            waiter = self.responses.get(message['id'])
                        if waiter is not None:
                            waiter.put(message)
                        continue
                    method = message.get('method', '')
                    params = message.get('params') or {}
                    is_root = params.get('threadId') == self.root_id
                    if method == 'turn/started' and is_root:
                        self.active_turn = params.get('turn', {}).get('id')
                    elif method == 'turn/completed' and is_root:
                        self.active_turn = None
                    if method == 'thread/goal/updated' and is_root:
                        write_json(self.directory / 'goal.json', params)
                    if 'id' in message:
                        # The native TUI answers server requests on its connection.
                        # An observer must not race it with a rejection or invented answer.
                        self.pending[message['id']] = message
                        write_json(self.directory / 'pending-inputs.json', list(self.pending.values()))
                    elif method == 'serverRequest/resolved':
                        self.pending.pop(params.get('requestId'), None)
                        write_json(self.directory / 'pending-inputs.json', list(self.pending.values()))
                    if method in ('item/started', 'item/completed', 'turn/started', 'turn/completed', 'error'):
                        self.latest = {'observed_at': time.time(), 'method': method, 'params': params}
                        if is_root:
                            write_json(self.directory / 'activity.json', self.latest)
                        thread = params.get('threadId')
                        if thread and all(c in '0123456789abcdef-' for c in thread):
                            write_json(self.directory / 'agents' / thread / 'activity.json', self.latest)
        except (ConnectionClosed, OSError) as error:
            write_json(self.directory / 'transport-error.json', {'error': str(error)})
        finally:
            with self.lock:
                self.closed = True
                for ident, waiter in self.responses.items():
                    waiter.put({'id': ident, 'error': {'message': 'Native app-server connection closed'}})

    def send(self, message):
        with self.send_lock:
            self.connection.send(json.dumps(message, ensure_ascii=False))

    def call(self, method, params, timeout=240):
        with self.lock:
            if self.closed:
                raise RuntimeError('Native app-server connection is closed')
            self.next_id += 1
            ident = self.next_id
            waiter = self.responses[ident] = queue.Queue()
        try:
            self.send({'id': ident, 'method': method, 'params': params})
            response = waiter.get(timeout=timeout)
        finally:
            with self.lock:
                self.responses.pop(ident, None)
        if 'error' in response:
            raise RuntimeError(f'{method}: {response["error"]}')
        return response.get('result', {})

    def close(self):
        with contextlib.suppress(ConnectionClosed, OSError):
            self.connection.close()
        self.reader.join(timeout=5)
