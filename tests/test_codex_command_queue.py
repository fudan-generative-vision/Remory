from contextlib import contextmanager
import json
import threading
import time

import pytest
from websockets.sync.client import unix_connect
from websockets.sync.server import unix_serve

from remory.codex.frontend import endpoint


class Controls:
    def __init__(self):
        self.executed, self.selected = [], []
        self.selecting, self.release = threading.Event(), threading.Event()

    def candidates(self):
        return [dict(id='a'*64, label='Same label', step=0, current=True),
                dict(id='b'*64, label='Same label', step=1, current=False)]

    def select(self, ident):
        self.selecting.set()
        assert self.release.wait(10), 'The test did not release checkpoint validation'
        self.selected.append(ident)
        return 'REMORY: selected fixture'

    def execute(self, args):
        self.executed.append(args[0])
        return 'REMORY: executed ' + args[0]


@contextmanager
def connection(tmp_path, control):
    seen = []
    def upstream(ws):
        for raw in ws:
            request = json.loads(raw)
            seen.append(request)
            ws.send(json.dumps({'id': request['id'], 'result': {'echo': True}}))
    path = tmp_path / 'upstream.sock'
    with unix_serve(upstream, str(path), compression=None) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with endpoint(tmp_path/'session', 'unix://'+str(path), control=control) as address:
                with unix_connect(address.removeprefix('unix://'), compression=None) as ws:
                    yield ws, seen
        finally:
            control.release.set()
            server.shutdown()
            worker.join(timeout=5)


def send(ws, ident, text, method='turn/steer'):
    ws.send(json.dumps(dict(id=ident, method=method, params=dict(threadId='root',
        expectedTurnId='real-turn', clientUserMessageId='client-'+str(ident),
        input=[dict(type='text', text=text)]))))


def until(ws, events, predicate):
    deadline = time.monotonic() + 5
    while not predicate(events):
        assert time.monotonic() < deadline, events
        message = json.loads(ws.recv(timeout=5))
        assert 'error' not in message, message
        events.append(message)
    return events


def has_text(events, text):
    return any(text in e.get('params', {}).get('delta', '') for e in events)


@pytest.mark.parametrize('prefix', ['/', '\\'])
@pytest.mark.parametrize('method', ['turn/start', 'turn/steer', 'thread/queue/add'])
def test_busy_load_acknowledges_duplicates_queues_mutations_and_keeps_status_live(tmp_path, method, prefix):
    control = Controls()
    with connection(tmp_path, control) as (ws, seen):
        events = []
        try:
            send(ws, 1, prefix + 'remory load', method)
            until(ws, events, lambda es: any(e.get('method') == 'item/tool/requestUserInput' for e in es))
            picker = next(e for e in events if e.get('method') == 'item/tool/requestUserInput')
            options = picker['params']['questions'][0]['options']
            assert options[0]['label'] != options[1]['label']
            ws.send(json.dumps(dict(id=picker['id'], result={'answers': {
                'checkpoint': {'answers': [options[1]['label']]}}})))
            assert control.selecting.wait(5)
            until(ws, events, lambda es: has_text(es, 'verifying the selected checkpoint'))
            assert any(e.get('method') == 'serverRequest/resolved' for e in events)
            send(ws, 2, '\\remory load', method)
            send(ws, 3, prefix + 'remory status', method)
            send(ws, 4, prefix + 'remory stop', method)
            send(ws, 5, prefix + 'remory train', method)
            send(ws, 6, 'ordinary user message', method)
            until(ws, events, lambda es: has_text(es, 'repeated command was ignored')
                  and has_text(es, 'executed status') and has_text(es, 'queued stop')
                  and has_text(es, 'queued train') and any(e.get('id') == 6 for e in es))
            assert control.executed == ['status'] and not control.selected
            assert [r['id'] for r in seen] == [6]
            assert {e.get('id') for e in events} >= {1, 2, 3, 4, 5, 6}
            control.release.set()
            until(ws, events, lambda es: has_text(es, 'executed train'))
            assert control.selected == ['b'*64]
            assert control.executed == ['status', 'stop', 'train']
            assert sum(e.get('method') == 'item/tool/requestUserInput' for e in events) == 1
        finally:
            control.release.set()


def test_cancel_picker_releases_queued_work_and_late_answers_stay_local(tmp_path):
    control = Controls()
    with connection(tmp_path, control) as (ws, seen):
        events = []
        send(ws, 1, '\\remory load', 'turn/start')
        until(ws, events, lambda es: any(e.get('method') == 'item/tool/requestUserInput' for e in es))
        picker = next(e for e in events if e.get('method') == 'item/tool/requestUserInput')
        send(ws, 2, '\\remory stop')
        until(ws, events, lambda es: has_text(es, 'queued stop'))
        ws.send(json.dumps(dict(id=3, method='turn/interrupt', params=dict(
            threadId='root', turnId=picker['params']['turnId']))))
        until(ws, events, lambda es: has_text(es, 'selection cancelled')
              and has_text(es, 'executed stop') and any(e.get('id') == 3 for e in es))
        assert not control.selected and not seen
        ws.send(json.dumps(dict(id=picker['id'], result={'answers': {
            'checkpoint': {'answers': ['Same label · bbbbbbbb']}}})))
        ws.send(json.dumps(dict(id=4, method='turn/interrupt', params=dict(
            threadId='root', turnId='real-model-turn'))))
        until(ws, events, lambda es: any(e.get('id') == 4 for e in es))
        assert [r['id'] for r in seen] == [4]
