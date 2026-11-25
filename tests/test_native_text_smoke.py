"""Optional real-native smoke tests; all model responses and state stay local.

Run with R_CODEX_TEST_BINARY=/absolute/path/to/codex. No model, API account,
research service, or GPU is used. The tests exercise the pinned native 0.149
app-server RPC contract rather than substituting a mock native client.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

import pytest


SUMMARY = 'TEXT_SMOKE_SUMMARY: native-smoke.txt contains native-tool-ok; keep this fact.'


class ChatStub:
    """A deterministic Chat Completions HTTP server, including real tool calls."""

    def __init__(self):
        self.requests = []
        self.errors = []
        self.summary_count = 0
        self.next_prompt_tokens = 64

    def answer(self, body):
        assert body['model'] == 'deterministic-test-model'
        assert body.get('stream', False) is False
        self.requests.append(body)
        assert len(self.requests) < 40, 'Unexpected native retry/tool loop'
        messages = body['messages']
        if not body.get('tools'):
            self.summary_count += 1
            message = {'role': 'assistant', 'content': SUMMARY}
        else:
            last_user = next(m['content'] for m in reversed(messages) if m['role'] == 'user')
            if 'SMOKE_SPAWN' in last_user:
                names = [tool['function']['name'] for tool in body['tools']]
                if messages[-1]['role'] != 'tool':
                    tool, ident = 'spawn_agent', 'call_native_spawn'
                    arguments = {'message': 'SMOKE_CHILD: respond with CHILD_SMOKE_OK.',
                                 'fork_context': False}
                elif messages[-1]['tool_call_id'] == 'call_native_spawn':
                    spawned = json.loads(messages[-1]['content'])
                    tool, ident = 'wait_agent', 'call_native_wait'
                    arguments = {'targets': [spawned['agent_id']], 'timeout_ms': 10000}
                else:
                    assert 'CHILD_SMOKE_OK' in messages[-1]['content']
                    tool = None
                    message = {'role': 'assistant', 'content': 'NATIVE_SMOKE_OK'}
                if tool:
                    name = next(name for name in names if name.split('__')[-1] == tool)
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                        'id': ident, 'type': 'function', 'function': {
                            'name': name, 'arguments': json.dumps(arguments)}}]}
            elif 'SMOKE_CHILD' in last_user:
                message = {'role': 'assistant', 'content': 'CHILD_SMOKE_OK'}
            elif 'SMOKE_EXECUTE' in last_user and messages[-1]['role'] != 'tool':
                names = [tool['function']['name'] for tool in body['tools']]
                name = next(name for name in names if name.split('__')[-1] == 'exec_command')
                command = ('python3 -c "from pathlib import Path; '
                           "p = Path('native-smoke.txt'); "
                           "p.write_text('native-tool-ok\\n'); print(p.read_text(), end='')\"")
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'call_native_smoke', 'type': 'function', 'function': {
                        'name': name, 'arguments': json.dumps({
                            'cmd': command, 'login': False, 'max_output_tokens': 1000})}}]}
            else:
                if 'SMOKE_EXECUTE' in last_user:
                    assert messages[-1]['role'] == 'tool'
                    assert 'native-tool-ok' in messages[-1]['content']
                if any(marker in last_user for marker in ['SMOKE_AFTER_COMPACT', 'SMOKE_RESUME']):
                    assert any(SUMMARY in str(m.get('content', '')) for m in messages)
                message = {'role': 'assistant', 'content': 'NATIVE_SMOKE_OK'}
        prompt_tokens, self.next_prompt_tokens = self.next_prompt_tokens, 64
        return {'id': 'chatcmpl-smoke-' + str(len(self.requests)),
                'object': 'chat.completion', 'created': 1, 'model': body['model'],
                'choices': [{'index': 0, 'message': message,
                             'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
                'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 12,
                          'total_tokens': prompt_tokens + 12}}

    @contextmanager
    def serve(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    assert self.path == '/v1/chat/completions'
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    result = stub.answer(body)
                    status = 200
                except Exception as error:
                    stub.errors.append(repr(error))
                    result, status = {'error': {'message': str(error)}}, 500
                encoded = json.dumps(result).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/v1'
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class NativePeer:
    """Synchronous RPC peer retaining notifications needed for assertions."""

    def __init__(self, address):
        from websockets.sync.client import unix_connect
        self.connection = unix_connect(address.removeprefix('unix://'), compression=None,
                                       max_size=None, open_timeout=10, ping_timeout=None)
        self.events = []
        self.next_id = 0
        self.call('initialize', {'clientInfo': {'name': 'text-native-smoke', 'version': '1'},
                                 'capabilities': {'experimentalApi': True}})
        self.send({'method': 'initialized'})

    def send(self, message):
        self.connection.send(json.dumps(message))

    def receive(self, timeout):
        message = json.loads(self.connection.recv(timeout=timeout))
        self.events.append(message)
        assert not ('method' in message and 'id' in message), (
            'Unexpected native approval/input request: ' + str(message))
        return message

    def call(self, method, params):
        self.next_id += 1
        ident = self.next_id
        self.send({'id': ident, 'method': method, 'params': params})
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            message = self.receive(max(.1, deadline - time.monotonic()))
            if message.get('id') == ident:
                assert 'error' not in message, message
                return message.get('result', {})
        pytest.fail(f'Timed out calling {method}: {self.events[-3:]}')

    def wait(self, method, *, since=0, predicate=lambda _params: True):
        deadline = time.monotonic() + 45
        while True:
            for event in self.events[since:]:
                if event.get('method') == method and predicate(event.get('params', {})):
                    return event['params']
            since = len(self.events)
            if time.monotonic() >= deadline:
                pytest.fail(f'Timed out waiting for {method}: {self.events[-3:]}')
            self.receive(max(.1, deadline - time.monotonic()))

    def turn(self, thread_id, text):
        since = len(self.events)
        result = self.call('turn/start', {'threadId': thread_id,
                                        'input': [{'type': 'text', 'text': text}]})
        turn_id = result['turn']['id']
        completed = self.wait('turn/completed', since=since,
                              predicate=lambda p: p['turn']['id'] == turn_id)
        assert completed['turn']['status'] == 'completed', completed
        assert completed['turn'].get('error') is None, completed
        return self.events[since:]

    def close(self):
        self.connection.close()


@pytest.fixture
def native_binary():
    value = os.environ.get('R_CODEX_TEST_BINARY')
    if not value:
        pytest.skip('Set R_CODEX_TEST_BINARY to the pinned Codex 0.149 binary')
    binary = Path(value).resolve()
    assert binary.is_file() and os.access(binary, os.X_OK), binary
    return binary


class Runtime:
    def __init__(self, directory, binary):
        from r_codex.cli import build_catalog
        self.directory = directory
        self.binary = binary
        self.workspace = directory / 'workspace'
        self.session = directory / 'session'
        self.workspace.mkdir()
        (self.session / '.codex').mkdir(parents=True)
        home = directory / 'isolated-home'
        home.mkdir()
        # Deliberately do not inherit credentials, shared Codex homes, MCP
        # configuration, proxies, or shell startup configuration from the caller.
        self.env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'LANG': 'C.UTF-8',
                    'CODEX_HOME': str(self.session / '.codex'),
                    'XDG_CONFIG_HOME': str(home / '.config'),
                    'XDG_DATA_HOME': str(home / '.local/share'),
                    'XDG_CACHE_HOME': str(home / '.cache'),
                    'R_CODEX_SESSION_TOKEN': 'native-test-only',
                    'NO_PROXY': '*', 'no_proxy': '*'}
        self.catalog = build_catalog(binary, self.session, self.env, 32768,
                                     'Deterministic local test')

    @contextmanager
    def start(self, url, *, thread_id=None, auto_limit=None):
        from r_codex.cli import MODEL_SLUG, runtime_config, serve, toml
        from r_codex.host import host
        from r_codex.provider import Provider
        provider = Provider(url, 'deterministic-test-model', self.session / 'provider',
                            session_id='native-smoke-session', max_new_tokens=512)
        with serve(provider, 'native-test-only') as bridge:
            config = runtime_config(bridge, self.catalog, self.workspace, 32768, 512)
            if auto_limit is not None:
                config['model_auto_compact_token_limit'] = auto_limit
            config.update(approval_policy='never', sandbox_mode='danger-full-access')
            config['shell_environment_policy'] = {'inherit': 'all', 'ignore_default_excludes': True}
            (self.session / '.codex/config.toml').write_text(
                '\n'.join(f'{key} = {toml(value)}' for key, value in config.items()) + '\n')
            with host(self.session, self.binary, self.workspace, self.env,
                      thread_id, MODEL_SLUG, unsafe=True) as info:
                peer = NativePeer(info['endpoint'])
                try:
                    # Subscribes this peer to the durable thread created by host.
                    peer.call('thread/resume', {'threadId': info['thread_id'],
                                               'excludeTurns': True})
                    yield peer, info['thread_id']
                finally:
                    peer.close()


def test_native_tool_manual_compaction_and_restart_resume(tmp_path, native_binary):
    runtime = Runtime(tmp_path, native_binary)
    stub = ChatStub()
    with stub.serve() as url:
        with runtime.start(url) as (peer, thread_id):
            events = peer.turn(thread_id, 'SMOKE_EXECUTE: create and read the test file.')
            assert (runtime.workspace / 'native-smoke.txt').read_text() == 'native-tool-ok\n'
            commands = [event['params']['item'] for event in events
                        if event.get('method') == 'item/completed'
                        and event['params']['item']['type'] == 'commandExecution']
            assert commands and commands[-1]['exitCode'] == 0
            assert 'native-tool-ok' in commands[-1]['aggregatedOutput']
            since = len(peer.events)
            assert peer.call('thread/compact/start', {'threadId': thread_id}) == {}
            compacted = peer.wait('turn/completed', since=since,
                                  predicate=lambda p: p['threadId'] == thread_id)
            assert compacted['turn']['status'] == 'completed', compacted
            assert stub.summary_count == 1
            peer.turn(thread_id, 'SMOKE_AFTER_COMPACT: recall the saved fact.')
        # Both native app-server and Provider are recreated. The native rollout
        # must carry the summary across this boundary without an in-memory handle.
        with runtime.start(url, thread_id=thread_id) as (peer, resumed_id):
            assert resumed_id == thread_id
            peer.turn(thread_id, 'SMOKE_RESUME: recall the saved fact again.')
    assert not stub.errors, stub.errors
    assert stub.summary_count == 1
    assert len(stub.requests) == 5


def test_native_automatic_text_compaction(tmp_path, native_binary):
    runtime = Runtime(tmp_path, native_binary)
    stub = ChatStub()
    with stub.serve() as url, runtime.start(url, auto_limit=10000) as (peer, thread_id):
        # Deterministic usage, not a huge prompt or real model, crosses native's
        # actual automatic-compaction threshold on the following turn.
        stub.next_prompt_tokens = 11000
        peer.turn(thread_id, 'SMOKE_PRIME_AUTO: remember a fact.')
        peer.turn(thread_id, 'SMOKE_AFTER_COMPACT: verify automatic text compaction.')
        assert stub.summary_count == 1
        assert any(event.get('method') == 'item/completed'
                   and event['params']['item']['type'] == 'contextCompaction'
                   for event in peer.events)
    assert not stub.errors, stub.errors


def test_native_child_agent_turn(tmp_path, native_binary):
    runtime = Runtime(tmp_path, native_binary)
    stub = ChatStub()
    with stub.serve() as url, runtime.start(url) as (peer, thread_id):
        events = peer.turn(thread_id, 'SMOKE_SPAWN: delegate one finite greeting and wait for it.')
        spawned = [event['params']['item'] for event in events
                   if event.get('method') == 'item/completed'
                   and event['params']['item']['type'] == 'collabAgentToolCall'
                   and event['params']['item']['tool'] == 'spawnAgent']
        assert len(spawned) == 1 and spawned[0]['status'] == 'completed', spawned
        child_ids = spawned[0]['receiverThreadIds']
        assert len(child_ids) == 1 and child_ids[0] != thread_id
        child = peer.call('thread/read', {'threadId': child_ids[0], 'includeTurns': True})
        assert child['thread']['turns'][-1]['status'] == 'completed', child
        assert any(item.get('text') == 'CHILD_SMOKE_OK'
                   for item in child['thread']['turns'][-1]['items'])
    assert not stub.errors, stub.errors
    assert len(stub.requests) == 4
