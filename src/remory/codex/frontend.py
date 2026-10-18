"""Local REMORY commands on the native Codex transport, before model input."""
from contextlib import contextmanager
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone

from websockets.sync.client import unix_connect
from websockets.sync.server import unix_serve
from websockets.exceptions import ConnectionClosed

from .transport import endpoint_path
from ..online.control import command


class Connection:
    def __init__(self, client, upstream, control):
        self.client, self.upstream, self.control = client, upstream, control
        self.lock, self.busy = threading.Lock(), threading.Lock()
        self.answers = {}
        self.active_turns = {}
        self.turn = None
        self.closed = threading.Event()

    def send(self, value):
        # Native 0.149 requires lifecycle timestamps. Without them it silently
        # drops local item acknowledgements, leaving steering input "pending".
        timestamp_field = {'item/started': 'startedAtMs',
                           'item/completed': 'completedAtMs'}.get(value.get('method'))
        if timestamp_field and timestamp_field not in value.get('params', {}):
            now = int(time.time() * 1000)
            value = dict(value, emittedAtMs=now,
                         params=dict(value['params'], **{timestamp_field: now}))
        with self.lock:
            self.client.send(json.dumps(value, ensure_ascii=False))

    def upstream_messages(self):
        try:
            for raw in self.upstream:
                message = json.loads(raw)
                params = message.get('params') or {}
                if message.get('method') == 'turn/started':
                    self.active_turns[params['threadId']] = params['turn']['id']
                elif message.get('method') == 'turn/completed':
                    self.active_turns.pop(params['threadId'], None)
                self.send(message)
        except ConnectionClosed:
            pass
        finally:
            self.closed.set()
            self.client.close()

    def run(self):
        relay = threading.Thread(target=self.upstream_messages, daemon=True)
        relay.start()
        try:
            for raw in self.client:
                message = json.loads(raw)
                ident = message.get('id')
                if 'method' not in message and ident in self.answers:
                    self.answers[ident].put(message)
                    continue
                if 'method' not in message and isinstance(ident, str) and ident.startswith('remory-pick-'):
                    continue  # A late answer to a cancelled local picker.
                args = command(message.get('params') or {}) if message.get('method') in ('turn/start', 'turn/steer', 'thread/queue/add') else None
                if args is not None:
                    if self.busy.acquire(blocking=False):
                        threading.Thread(target=self.local, args=(message, args), daemon=True).start()
                    else:
                        self.send(dict(id=ident, error=dict(code=-32600, message='A REMORY command is already open.')))
                elif message.get('method') == 'turn/interrupt' and (message.get('params') or {}).get('turnId') == self.turn:
                    for waiter in list(self.answers.values()):
                        waiter.put({'result': {'answers': {}}})
                    self.send(dict(id=ident, result={}))
                else:
                    self.upstream.send(raw)
        finally:
            self.closed.set()
            self.upstream.close()
            for waiter in list(self.answers.values()):
                waiter.put({'result': {'answers': {}}})

    def local(self, request, args):
        params = request['params']
        thread = params['threadId']
        queued = request['method'] == 'thread/queue/add'
        steering = request['method'] != 'turn/start'
        turn = (params.get('expectedTurnId') or self.active_turns.get(thread) or str(uuid.uuid4())) if steering else str(uuid.uuid4())
        self.turn = turn
        value = dict(id=turn, items=[], status='inProgress', error=None)
        try:
            if queued:
                self.send(dict(id=request['id'], result={'queuedSubmission': dict(id=str(uuid.uuid4()),
                    clientUserMessageId=params['clientUserMessageId'], input=params['input'])}))
                self.send(dict(method='thread/queue/changed', params=dict(threadId=thread)))
            else:
                self.send(dict(id=request['id'], result={'turnId': turn} if steering else {'turn': value}))
            if not steering:
                self.send(dict(method='turn/started', params=dict(threadId=thread, turn=value)))
            else:
                # Busy TUI input uses turn/steer and keeps a local pending
                # message until it sees the native userMessage acknowledgement.
                # The command is consumed here, never by the model.
                consumed = dict(type='userMessage', id='remory-command-' + uuid.uuid4().hex,
                                clientId=params.get('clientUserMessageId'), content=params['input'])
                self.send(dict(method='item/started', params=dict(threadId=thread,
                    turnId=turn, item=consumed)))
                self.send(dict(method='item/completed', params=dict(threadId=thread,
                    turnId=turn, item=consumed)))
            result = self.pick(thread, turn) if args == ['load'] else self.control.execute(args)
            item = dict(type='agentMessage', id='remory-status-' + uuid.uuid4().hex, text=result, phase='commentary')
            self.send(dict(method='item/started', params=dict(threadId=thread, turnId=turn, item=item)))
            self.send(dict(method='item/agentMessage/delta', params=dict(threadId=thread,
                turnId=turn, itemId=item['id'], delta=result)))
            self.send(dict(method='item/completed', params=dict(threadId=thread, turnId=turn, item=item)))
        except Exception as exc:
            item = dict(type='agentMessage', id='remory-error-' + uuid.uuid4().hex,
                        text='REMORY: ' + str(exc), phase='commentary')
            if not self.closed.is_set():
                self.send(dict(method='item/agentMessage/delta', params=dict(threadId=thread,
                    turnId=turn, itemId=item['id'], delta=item['text'])))
                self.send(dict(method='item/completed', params=dict(threadId=thread, turnId=turn, item=item)))
        finally:
            if not steering and not self.closed.is_set():
                value['status'] = 'completed'
                self.send(dict(method='turn/completed', params=dict(threadId=thread, turn=value)))
            self.turn = None
            self.busy.release()

    def pick(self, thread, turn):
        options = self.control.candidates()
        if not options:
            return 'REMORY: no complete compatible checkpoints are available.'
        labels = {x['label'] + ' · ' + x['id'][:8] + (' [current]' if x['current'] else ''): x['id']
                  for i, x in enumerate(options)}
        request_id = 'remory-pick-' + uuid.uuid4().hex
        waiter = self.answers[request_id] = queue.Queue()
        entries = [dict(label=label, description=f"Step {x['step'] if x['step'] is not None else 'base'} · " +
                       (datetime.fromtimestamp(x['created'], timezone.utc).strftime('%m-%d %H:%M UTC') + ' · ' if x.get('created') else '') + x['id'][:12])
                   for label, x in zip(labels, options)]
        entries.append(dict(label='Cancel', description='Keep the current checkpoint.'))
        self.send(dict(id=request_id, method='item/tool/requestUserInput', params=dict(
            threadId=thread, turnId=turn, itemId=request_id, isBlocking=True,
            questions=[dict(id='checkpoint', header='REMORY', question='Load a checkpoint',
                            isOther=False, isSecret=False, options=entries)])))
        try:
            response = waiter.get()
            answers = (response.get('result') or {}).get('answers', {}).get('checkpoint', {}).get('answers', [])
            if len(answers) != 1 or answers[0] not in labels:
                return 'REMORY: checkpoint selection cancelled.'
            return self.control.select(labels[answers[0]])
        finally:
            self.answers.pop(request_id, None)
            if not self.closed.is_set():
                self.send(dict(method='serverRequest/resolved', params=dict(threadId=thread, requestId=request_id)))


@contextmanager
def endpoint(session, upstream, *, control):
    # Each TUI gets its own proxy; native observers and other clients stay untouched.
    path = endpoint_path(session).with_name('remory-' + uuid.uuid4().hex[:10] + '.sock')
    clients = set()
    clients_lock = threading.Lock()
    def handle(client):
        with clients_lock:
            clients.add(client)
        try:
            with unix_connect(upstream.removeprefix('unix://'), compression=None,
                              max_size=None, ping_timeout=None) as remote:
                Connection(client, remote, control).run()
        except (ConnectionClosed, OSError):
            pass
        finally:
            with clients_lock:
                clients.discard(client)
    with unix_serve(handle, str(path), compression=None, max_size=None, ping_timeout=None) as server:
        os.chmod(path, 0o600)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield 'unix://' + str(path)
        finally:
            with clients_lock:
                active = list(clients)
            for client in active:
                client.close()
            server.shutdown()
            worker.join(timeout=5)
            path.unlink(missing_ok=True)
