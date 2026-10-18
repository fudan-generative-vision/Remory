import json
from pathlib import Path
import tempfile
import threading
import unittest

from websockets.sync.client import unix_connect
from websockets.sync.server import unix_serve

from remory.online import control
from remory.codex.frontend import endpoint


class CommandsTest(unittest.TestCase):
    def test_only_standalone_text_is_a_command(self):
        def parse(text, *rest):
            return control.command({'input': [{'type': 'text', 'text': text}, *rest]})
        for prefix in ('/', '\\'):
            self.assertEqual(parse(prefix + 'remory train'), ['train'])
            self.assertEqual(parse(prefix + 'remory load'), ['load'])
            self.assertEqual(parse(' ' + prefix + 'remory status '), ['status'])
            self.assertEqual(parse(prefix + 'remory'), ['help'])
            self.assertIsNone(parse('Please explain ' + prefix + 'remory train'))
            self.assertIsNone(parse('`' + prefix + 'remory train`'))
            self.assertIsNone(parse(prefix + 'remory train', {'type': 'image', 'imageUrl': 'data:...'}))
            self.assertIsNone(parse(prefix + 'remoryx train'))
        self.assertIsNone(control.command({'input': [None]}))
        self.assertIsNone(parse('/compact'))

    def test_native_transport_picker_and_input_isolation(self):
        seen, selected = [], []
        class Stub:
            def candidates(self):
                return [dict(id='a'*64, label='base', step=0, current=True),
                        dict(id='b'*64, label='trained', step=10, current=False)]
            def select(self, ident):
                selected.append(ident)
                return 'REMORY loaded'
            def execute(self, args):
                return 'REMORY collection started'
        def echo(ws):
            for raw in ws:
                msg = json.loads(raw)
                seen.append(msg)
                ws.send(json.dumps(dict(id=msg['id'], result={'echo': True})))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'native.sock'
            with unix_serve(echo, str(path), compression=None) as server:
                runner = threading.Thread(target=server.serve_forever, daemon=True)
                runner.start()
                with endpoint(Path(tmp) / 'session', 'unix://' + str(path), control=Stub()) as address:
                    with unix_connect(address.removeprefix('unix://'), compression=None) as ws:
                        ws.send(json.dumps(dict(id=1, method='turn/start', params=dict(threadId='thread',
                            input=[{'type': 'text', 'text': '/remory load'}]))))
                        notifications = []
                        while True:
                            msg = json.loads(ws.recv(timeout=5))
                            notifications.append(msg)
                            if msg.get('method') == 'item/tool/requestUserInput':
                                self.assertEqual(msg['params']['questions'][0]['options'][1]['label'], 'trained · bbbbbbbb')
                                ws.send(json.dumps(dict(id=msg['id'], result={'answers': {'checkpoint': {'answers': ['trained · bbbbbbbb']}}})))
                            if msg.get('method') == 'turn/completed':
                                break
                        self.assertEqual(selected, ['b'*64])
                        self.assertEqual(seen, [])
                        ws.send(json.dumps(dict(id=2, method='turn/start', params=dict(threadId='thread',
                            input=[{'type': 'text', 'text': 'ordinary user message'}]))))
                        self.assertEqual(json.loads(ws.recv(timeout=5))['result'], {'echo': True})
                        self.assertEqual(len(seen), 1)
                server.shutdown()
                runner.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
