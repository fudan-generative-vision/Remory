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
        self.lock, self.state_lock = threading.Lock(), threading.Lock()
        self.jobs = queue.Queue()
        self.pending = set()
        self.answers = {}
        self.picker_turns = {}
        self.active_turns = {}
        self.local_turns = set()
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
        threading.Thread(target=self.work, daemon=True).start()
        try:
            for raw in self.client:
                message = json.loads(raw)
                ident = message.get('id')
                with self.state_lock:
                    waiter = self.answers.get(ident) if 'method' not in message else None
                if waiter is not None:
                    waiter.put(message)
                    continue
                if 'method' not in message and isinstance(ident, str) and ident.startswith('remory-pick-'):
                    continue  # A late answer to a cancelled local picker.
                args = command(message.get('params') or {}) if message.get('method') in ('turn/start', 'turn/steer', 'thread/queue/add') else None
                if args is not None:
                    self.accept(message, args)
                elif message.get('method') == 'turn/interrupt' and self.cancel_picker(message):
                    self.send(dict(id=ident, result={}))
                else:
                    self.upstream.send(raw)
        finally:
            self.closed.set()
            self.upstream.close()
            self.jobs.put(None)
            with self.state_lock:
                waiters = list(self.answers.values())
            for waiter in waiters:
                waiter.put({'result': {'answers': {}}})

    def cancel_picker(self, request):
        turn = (request.get('params') or {}).get('turnId')
        with self.state_lock:
            waiters = [self.answers[key] for key, value in self.picker_turns.items()
                       if value == turn]
            local = turn in self.local_turns
        for waiter in waiters:
            waiter.put({'result': {'answers': {}}})
        return local or bool(waiters)

    def accept(self, request, args):
        # Always acknowledge local input as a successful transport operation.
        # A duplicate command is UI state, not a fatal turn/steer RPC error.
        params = request['params']
        thread = params['threadId']
        queued = request['method'] == 'thread/queue/add'
        steering = request['method'] != 'turn/start'
        turn = (params.get('expectedTurnId') or self.active_turns.get(thread) or str(uuid.uuid4())) if steering else str(uuid.uuid4())
        value = dict(id=turn, items=[], status='inProgress', error=None)
        context = (thread, turn, value, steering)
        if queued:
            self.send(dict(id=request['id'], result={'queuedSubmission': dict(id=str(uuid.uuid4()),
                clientUserMessageId=params['clientUserMessageId'], input=params['input'])}))
            self.send(dict(method='thread/queue/changed', params=dict(threadId=thread)))
        else:
            self.send(dict(id=request['id'], result={'turnId': turn} if steering else {'turn': value}))
        if not steering:
            with self.state_lock:
                self.local_turns.add(turn)
            self.send(dict(method='turn/started', params=dict(threadId=thread, turn=value)))
        else:
            consumed = dict(type='userMessage', id='remory-command-' + uuid.uuid4().hex,
                            clientId=params.get('clientUserMessageId'), content=params['input'])
            self.send(dict(method='item/started', params=dict(threadId=thread,
                turnId=turn, item=consumed)))
            self.send(dict(method='item/completed', params=dict(threadId=thread,
                turnId=turn, item=consumed)))
        if args in (['status'], ['help']):
            # Read-only commands remain available during checkpoint validation.
            threading.Thread(target=self.local, args=(context, args), daemon=True).start()
            return
        key = tuple(args)
        with self.state_lock:
            duplicate, waiting = key in self.pending, bool(self.pending)
            if not duplicate:
                self.pending.add(key)
        if duplicate:
            self.say(context, 'REMORY: ' + ' '.join(args) +
                     ' is already in progress; the repeated command was ignored.')
            self.complete(context)
            return
        if waiting:
            self.say(context, 'REMORY: queued ' + ' '.join(args) +
                     '; waiting for the current command to finish.')
        self.jobs.put((context, args))

    def work(self):
        while not self.closed.is_set():
            job = self.jobs.get()
            if job is None or self.closed.is_set():
                return
            context, args = job
            try:
                self.local(context, args)
            finally:
                with self.state_lock:
                    self.pending.discard(tuple(args))

    def say(self, context, text):
        if self.closed.is_set():
            return
        thread, turn, _, _ = context
        item = dict(type='agentMessage', id='remory-status-' + uuid.uuid4().hex, text=text, phase='commentary')
        self.send(dict(method='item/started', params=dict(threadId=thread, turnId=turn, item=item)))
        self.send(dict(method='item/agentMessage/delta', params=dict(threadId=thread,
            turnId=turn, itemId=item['id'], delta=text)))
        self.send(dict(method='item/completed', params=dict(threadId=thread, turnId=turn, item=item)))

    def complete(self, context):
        thread, turn, value, steering = context
        if not steering:
            with self.state_lock:
                self.local_turns.discard(turn)
            if not self.closed.is_set():
                value['status'] = 'completed'
                self.send(dict(method='turn/completed', params=dict(threadId=thread, turn=value)))

    def local(self, context, args):
        thread, turn, _, _ = context
        try:
            result = self.pick(thread, turn) if args == ['load'] else self.control.execute(args)
            self.say(context, result)
        except Exception as exc:
            self.say(context, 'REMORY: ' + str(exc))
        finally:
            self.complete(context)

    def pick(self, thread, turn):
        options = self.control.candidates()
        if not options:
            return 'REMORY: no complete compatible checkpoints are available.'
        labels = {x['label'] + ' · ' + x['id'][:8] + (' [current]' if x['current'] else ''): x['id']
                  for i, x in enumerate(options)}
        request_id = 'remory-pick-' + uuid.uuid4().hex
        with self.state_lock:
            waiter = self.answers[request_id] = queue.Queue()
            self.picker_turns[request_id] = turn
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
        finally:
            with self.state_lock:
                self.answers.pop(request_id, None)
                self.picker_turns.pop(request_id, None)
            if not self.closed.is_set():
                self.send(dict(method='serverRequest/resolved', params=dict(threadId=thread, requestId=request_id)))
        if self.closed.is_set():
            return 'REMORY: checkpoint selection cancelled.'
        self.say((thread, turn, None, True), 'REMORY: verifying the selected checkpoint…')
        return self.control.select(labels[answers[0]])


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
