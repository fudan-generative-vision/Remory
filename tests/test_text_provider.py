import base64
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import httpx
import pytest

from r_codex.protocol import chat_output, messages_from_wire, response_events, tools_from_wire
from r_codex.provider import COMPACT_PROMPT, Provider, create_app


TOOL = {'type': 'function', 'name': 'exec_command', 'description': 'Run a shell command',
        'parameters': {'type': 'object', 'properties': {'cmd': {'type': 'string'}},
                       'required': ['cmd'], 'additionalProperties': False}}


def completion(text='Done.', *, tools=None, reason='stop', reasoning=None):
    message = {'role': 'assistant', 'content': text}
    if tools is not None:
        message['tool_calls'] = tools
    if reasoning is not None:
        message['reasoning_content'] = reasoning
    return {'choices': [{'index': 0, 'message': message, 'finish_reason': reason}],
            'usage': {'prompt_tokens': 101, 'completion_tokens': 12,
                      'prompt_tokens_details': {'cached_tokens': 34}}}


def function(name='exec_command', args=None, *, call_id='backend_call'):
    return {'id': call_id, 'type': 'function', 'function': {
        'name': name, 'arguments': json.dumps(args if args is not None else {'cmd': 'pwd'})}}


def factory(tmp_path, backend, **kwargs):
    def make():
        return Provider('http://backend/v1', 'test-model', tmp_path, session_id='same',
                        transport=httpx.MockTransport(backend), **kwargs)
    return make


def test_structured_tools_namespaces_custom_and_reasoning_roundtrip():
    wire_tools = [TOOL, {'type': 'namespace', 'name': 'multi_agent_v1', 'tools': [
        {'type': 'function', 'name': 'spawn_agent', 'parameters': {
            'type': 'object', 'properties': {'message': {'type': 'string'},
                                           'fork_context': {'type': 'boolean'}}}}]},
        {'type': 'custom', 'name': 'apply_patch'}]
    original = deepcopy(wire_tools)
    native, routes = tools_from_wire(wire_tools)
    assert wire_tools == original
    assert native[1]['function']['parameters']['properties']['fork_context'] == {'type': 'boolean'}
    patch = '*** Begin Patch\n*** Add File: x\n+ok\n*** End Patch'
    message = {'content': 'Starting work.', 'reasoning_content': 'Choose the tools.',
               'tool_calls': [function('multi_agent_v1__spawn_agent',
                    {'message': 'Inspect files', 'fork_context': True}, call_id='child'),
                              function('apply_patch', {'input': patch}, call_id='patch')]}
    output = chat_output(message, routes)
    assert output[0]['type'] == 'reasoning'
    assert output[1]['channel'] == 'commentary'
    assert output[2]['namespace'] == 'multi_agent_v1'
    assert json.loads(output[2]['arguments'])['fork_context'] is True
    assert output[3]['type'] == 'custom_tool_call' and output[3]['input'] == patch
    back = messages_from_wire(output + [
        {'type': 'function_call_output', 'call_id': 'child', 'output': 'child started'},
        {'type': 'custom_tool_call_output', 'call_id': 'patch', 'output': 'applied'}])
    assert len(back) == 3
    assert back[0]['reasoning_content'] == 'Choose the tools.'
    assert back[0]['tool_calls'][0]['function']['name'] == 'multi_agent_v1__spawn_agent'
    assert isinstance(back[0]['tool_calls'][0]['function']['arguments'], str)
    assert json.loads(back[0]['tool_calls'][1]['function']['arguments']) == {'input': patch}
    assert back[1] == {'role': 'tool', 'tool_call_id': 'child', 'content': 'child started'}


@pytest.mark.parametrize('call, match', [
    (function('missing'), 'Unknown'),
    (function(args={'cmd': 42}), 'Invalid arguments'),
    (function(args=['not an object']), 'must be an object'),
])
def test_invalid_generated_tools_fail_closed(call, match):
    _, routes = tools_from_wire([TOOL])
    with pytest.raises(ValueError, match=match):
        chat_output({'tool_calls': [call]}, routes)


def test_zero_argument_tool_and_tool_result_pairing():
    _, routes = tools_from_wire([{'type': 'function', 'name': 'get_goal',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}])
    output = chat_output({'tool_calls': [function('get_goal', {})]}, routes)
    assert json.loads(output[0]['arguments']) == {}
    with pytest.raises(ValueError, match='Missing tool outputs'):
        messages_from_wire(output)
    with pytest.raises(ValueError, match='no pending call'):
        messages_from_wire([{'type': 'function_call_output', 'call_id': 'absent', 'output': 'x'}])
    with pytest.raises(ValueError, match='Duplicate generated'):
        chat_output({'tool_calls': [function('get_goal', {}), function('get_goal', {})]}, routes)


def test_images_remain_image_blocks_in_user_messages_and_tool_results():
    image = 'data:image/png;base64,aGVsbG8='
    messages = messages_from_wire([{'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'Inspect'},
        {'type': 'input_image', 'image_url': image, 'detail': 'high'}]},
        {'type': 'function_call', 'name': 'view_image', 'arguments': '{}', 'call_id': 'image'},
        {'type': 'function_call_output', 'call_id': 'image', 'output': [
            {'type': 'input_text', 'text': 'Image output'},
            {'type': 'input_image', 'image_url': image}]}])
    assert messages[0]['content'][1] == {'type': 'image_url',
        'image_url': {'url': image, 'detail': 'high'}}
    assert messages[2] == {'role': 'tool', 'tool_call_id': 'image', 'content': 'Image output'}
    assert messages[3]['role'] == 'user'
    assert messages[3]['content'][1]['image_url']['url'] == image


def test_unparsed_native_tools_fallback_and_unfinished_reasoning():
    _, routes = tools_from_wire([TOOL])
    output = chat_output({'content': '<think>Inspect.</think><tool_call>' + json.dumps({
        'name': 'exec_command', 'arguments': {'cmd': 'pwd'}}) + '</tool_call>'}, routes)
    assert output[0]['type'] == 'reasoning'
    assert json.loads(output[-1]['arguments']) == {'cmd': 'pwd'}
    for message in [{'content': '<think>Still working'},
                    {'content': '', 'reasoning_content': 'Still working'},
                    {'content': '<think>Just thinking.</think>'}]:
        with pytest.raises(ValueError, match='reasoning|no complete'):
            chat_output(message, routes)


def test_normal_generation_uses_only_chat_endpoint_and_preserves_tool_loop(tmp_path):
    received = []
    def backend(request):
        assert request.url.path == '/v1/chat/completions'
        body = json.loads(request.content)
        received.append(body)
        assert body['model'] == 'test-model' and body['stream'] is False
        assert 'input_ids' not in body and 'remory' not in body
        if len(received) == 1:
            return httpx.Response(200, json=completion('Inspecting.', tools=[function()],
                reason='tool_calls', reasoning='Use the shell.'))
        return httpx.Response(200, json=completion('The current directory is /work.'))
    make = factory(tmp_path, backend)
    request = {'model': 'text-model', 'tools': [TOOL], 'input': 'Where am I?',
               'reasoning': {'effort': 'medium'}}
    first = make().handle(request)
    assert 'reasoning_effort' not in received[0]
    assert first['model'] == 'text-model'
    assert first['usage'] == {'input_tokens': 101, 'output_tokens': 12, 'total_tokens': 113,
                              'input_tokens_details': {'cached_tokens': 34}}
    result = make().handle({**request, 'input': [{'role': 'user', 'content': 'Where am I?'}]
        + first['output'] + [{'type': 'function_call_output', 'call_id': 'backend_call',
                              'output': '/work'}]})
    assert result['output'][-1]['content'][0]['text'].endswith('/work.')
    assert received[-1]['messages'][-1] == {
        'role': 'tool', 'tool_call_id': 'backend_call', 'content': '/work'}


@pytest.mark.parametrize('force_compact', [False, True])
def test_compact_restart_retry_resume_and_second_compact(tmp_path, force_compact):
    received = []
    summaries = ['File x is complete; next inspect y.', 'Files x and y are complete.']
    def backend(request):
        assert request.url.path == '/v1/chat/completions'
        body = json.loads(request.content)
        received.append(body)
        if body['messages'][-1]['content'] == COMPACT_PROMPT:
            assert 'tools' not in body
            return httpx.Response(200, json=completion(summaries.pop(0)))
        assert 'File x is complete' in json.dumps(body['messages'])
        assert 'Raw old history' not in json.dumps(body['messages'])
        assert 'Never edit z.' in body['messages'][0]['content']
        return httpx.Response(200, json=completion('Now y is complete.'))
    make = factory(tmp_path, backend)
    history = [{'role': 'developer', 'content': 'Never edit z.'},
               {'role': 'user', 'content': 'Raw old history: finish x, then y.'}]
    request = {'instructions': 'Assist.', 'tools': [TOOL], 'input': history + (
        [] if force_compact else [{'type': 'compaction_trigger'}])}
    first = make().handle(request, force_compact=force_compact)
    assert first['object'] == ('response.compaction' if force_compact else 'response')
    marker = first['output'][0]['encrypted_content']
    assert marker.startswith('text.v1.')
    payload = json.loads(base64.urlsafe_b64decode(marker.removeprefix('text.v1.')))
    assert payload['summary'] == 'File x is complete; next inspect y.'
    assert 'handle' not in payload
    assert make().handle(request, force_compact=force_compact) == first
    assert len(received) == 1
    continuation = {'instructions': 'Assist.', 'tools': [TOOL],
                    'input': first['output'] + [{'role': 'user', 'content': 'Continue.'}]}
    answer = make().handle(continuation)
    second = make().handle({**continuation, 'input': continuation['input'] + answer['output']},
                           force_compact=True)
    assert len(received) == 3
    summary_request = received[-1]['messages']
    assert 'File x is complete' in json.dumps(summary_request)
    assert 'Now y is complete' in json.dumps(summary_request)
    second_payload = json.loads(base64.urlsafe_b64decode(
        second['output'][0]['encrypted_content'].removeprefix('text.v1.')))
    assert second_payload['summary'] == 'Files x and y are complete.'


@pytest.mark.parametrize('bad_result', [
    completion('unfinished', reason='length'), completion(''),
    completion('<think>unfinished'), completion('bad', tools=[function()], reason='tool_calls'),
    completion('<tool_call>{"name":"exec_command"}</tool_call>'),
])
def test_failed_compact_leaves_previous_state_unchanged(tmp_path, bad_result):
    outcomes = [completion('Prior useful summary.'), bad_result, completion('Still continuing.')]
    received = []
    def backend(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json=outcomes.pop(0))
    make = factory(tmp_path, backend)
    p = make()
    old = p.handle({'input': 'Original work'}, force_compact=True)
    path = p._path('root')
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='original context'):
        make().handle({'input': old['output'] + [{'role': 'user', 'content': 'Next work'}]},
                      force_compact=True)
    assert path.read_bytes() == before
    make().handle({'input': old['output'] + [{'role': 'user', 'content': 'Continue original'}]})
    assert 'Prior useful summary.' in json.dumps(received[-1]['messages'])


def test_transport_and_unfinished_generation_failures_do_not_save_a_response(tmp_path):
    for index, outcome in enumerate([httpx.Response(503, text='temporarily unavailable'),
                                    httpx.Response(200, json=completion('<think>unfinished'))]):
        p = factory(tmp_path / str(index), lambda request: outcome)()
        with pytest.raises((httpx.HTTPStatusError, ValueError)):
            p.handle({'input': 'Task'})
        assert not list(p.directory.glob('thread-*.json'))


def test_child_inherits_checkpoint_without_cross_thread_history_or_extra_calls(tmp_path):
    received = []
    def backend(request):
        body = json.loads(request.content)
        received.append(body)
        text = 'Shared parent summary.' if body['messages'][-1]['content'] == COMPACT_PROMPT else 'Done.'
        return httpx.Response(200, json=completion(text))
    make = factory(tmp_path, backend)
    root = {'thread-id': 'parent'}
    child = {'thread-id': 'child', 'x-codex-turn-metadata': json.dumps({
        'thread_id': 'child', 'parent_thread_id': 'parent'})}
    first = make().handle({'input': 'Parent original'}, root, force_compact=True)
    inherited = {'input': first['output'] + [{'role': 'user', 'content': 'Child secret task'}]}
    make().handle(inherited, child)
    assert len(received) == 2
    assert 'Shared parent summary.' in json.dumps(received[-1])
    make().handle({'input': first['output'] + [{'role': 'user', 'content': 'Parent continues'}]}, root)
    assert len(received) == 3
    assert 'Child secret task' not in json.dumps(received[-1])
    # Identical input under a different native thread must not replay another agent's response.
    make().handle(inherited, {'thread-id': 'grandchild', 'x-codex-turn-metadata': json.dumps({
        'thread_id': 'grandchild', 'parent_thread_id': 'child'})})
    assert len(received) == 4
    assert make().meta['root_thread_id'] == 'parent'


def test_checkpoint_rejection_and_session_identity(tmp_path):
    make = factory(tmp_path, lambda request: httpx.Response(200, json=completion('Summary.')))
    p = make()
    good = p.handle({'input': 'Task'}, force_compact=True)
    for marker in ['remory.v1.rm_old', 'text.v1.!!!', 'text.v1.' + base64.b64encode(b'[]').decode()]:
        with pytest.raises(ValueError, match='checkpoint'):
            p.handle({'input': [{'type': 'compaction', 'encrypted_content': marker}]})
    item = deepcopy(good['output'][0])
    payload = json.loads(base64.urlsafe_b64decode(item['encrypted_content'].removeprefix('text.v1.')))
    payload['identity'] = 'other backend'
    item['encrypted_content'] = 'text.v1.' + base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(ValueError, match='identity'):
        p.handle({'input': [item]})
    with pytest.raises(ValueError, match='model or URL changed'):
        Provider('http://backend/v1', 'different-model', tmp_path, session_id='same')
    with pytest.raises(ValueError, match='Session identity changed'):
        Provider('http://backend/v1', 'test-model', tmp_path, session_id='different')
    with pytest.raises(ValueError, match='previous_response_id'):
        p.handle({'previous_response_id': good['id'], 'input': 'Continue'})


def test_aborted_undelivered_tool_is_not_injected_into_next_request(tmp_path):
    received = []
    def backend(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json=(completion(None, tools=[function()], reason='tool_calls')
            if len(received) == 1 else completion('Cancelled.')))
    make = factory(tmp_path, backend)
    make().handle({'tools': [TOOL], 'input': 'Inspect project'})
    make().handle({'tools': [TOOL], 'input': [
        {'role': 'user', 'content': 'Inspect project'},
        {'role': 'user', 'content': '<turn_aborted>Interrupted</turn_aborted>'},
        {'role': 'user', 'content': 'Cancel that'}]})
    assert 'pwd' not in json.dumps(received[-1]['messages'])


def test_http_auth_stream_and_both_compact_routes(tmp_path):
    from fastapi.testclient import TestClient
    p = factory(tmp_path, lambda request: httpx.Response(200, json=completion('Hello.')))()
    client = TestClient(create_app(p, token='local'))
    assert client.get('/health').status_code == 401
    headers = {'Authorization': 'Bearer local'}
    assert client.get('/health', headers=headers).status_code == 200
    response = client.post('/v1/responses', headers=headers, json={'stream': True, 'input': 'Hello'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    assert 'event: response.completed' in response.text and 'Hello.' in response.text
    for route, data in [('/v1/responses/compact', {'input': 'First task', 'stream': True}),
                        ('/v1/responses', {'input': [{'role': 'user', 'content': 'Second task'},
                                                   {'type': 'compaction_trigger'}]})]:
        compact = client.post(route, headers=headers, json=data)
        assert compact.status_code == 200
        assert compact.json()['output'][0]['type'] == 'compaction'
    error = client.post('/v1/responses', headers=headers, json={
        'previous_response_id': 'not_supported', 'input': 'Continue'})
    assert error.status_code == 400


def test_sse_opening_items_do_not_duplicate_text_or_arguments():
    _, routes = tools_from_wire([TOOL, {'type': 'custom', 'name': 'apply_patch'}])
    output = chat_output({'content': 'Working.', 'tool_calls': [function(),
        function('apply_patch', {'input': 'patch'}, call_id='patch')]}, routes)
    events = list(response_events({'id': 'response_test', 'output': output, 'status': 'completed'}))
    added = [event['item'] for event in events if event['type'] == 'response.output_item.added']
    assert added[0]['content'] == []
    assert added[1]['arguments'] == ''
    assert added[2]['input'] == ''


def test_backend_key_and_reasoning_are_explicit_not_persisted(tmp_path):
    received = []
    def backend(request):
        assert request.headers['Authorization'] == 'Bearer private-key-for-test'
        received.append(json.loads(request.content))
        return httpx.Response(200, json=completion('Done.'))
    p = factory(tmp_path, backend, api_key='private-key-for-test', reasoning_effort='high')()
    p.handle({'input': 'Task', 'reasoning': {'effort': 'low'}})
    assert received[-1]['reasoning_effort'] == 'low'
    assert all('private-key-for-test' not in path.read_text() for path in tmp_path.glob('*.json'))


def test_provider_bypasses_ambient_proxies(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/v1/chat/completions'
            self.rfile.read(int(self.headers['Content-Length']))
            payload = json.dumps(completion('Direct local connection')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    for variable in ['ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy']:
        monkeypatch.setenv(variable, 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '')
    monkeypatch.setenv('no_proxy', '')
    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            p = Provider(f'http://127.0.0.1:{server.server_port}', 'test-model', tmp_path,
                         session_id='direct')
            result = p.handle({'input': 'Hello'})
            assert result['output'][0]['content'][0]['text'] == 'Direct local connection'
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_compaction_prompt_matches_pinned_public_local_prompt():
    assert hashlib.sha256(COMPACT_PROMPT.encode()).hexdigest() == (
        'ab0c334d4faca17e3afbb9b16967c1b2fdcc7242a9a0880af57949fa236d6d07')
