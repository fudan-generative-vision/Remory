import json

import httpx
import pytest

from remory.codex.protocol import messages_from_wire, parse_output, tools_from_wire
from remory.codex.provider import CodexChatTemplate, Provider, create_app

TOOL = {'type': 'function', 'name': 'exec_command', 'description': 'Run a shell command',
        'parameters': {'type': 'object', 'properties': {'cmd': {'type': 'string'}},
                       'required': ['cmd'], 'additionalProperties': False}}


def test_native_tools_preserve_namespaces_freeform_and_json_types():
    tools, routes = tools_from_wire([TOOL,
        {'type': 'namespace', 'name': 'multi_agent_v1', 'tools': [
            {'type': 'function', 'name': 'spawn_agent', 'parameters': {
                'type': 'object', 'properties': {'message': {'type': 'string'},
                                               'fork_context': {'type': 'boolean'}}}}]},
        {'type': 'custom', 'name': 'apply_patch'}])
    assert len(tools) == 3
    result = parse_output('Think.</think><tool_call><function=multi_agent_v1__spawn_agent>'
        '<parameter=message>Inspect file</parameter><parameter=fork_context>false</parameter>'
        '</function></tool_call>', routes)
    call = result[-1]
    assert call['namespace'] == 'multi_agent_v1'
    assert json.loads(call['arguments']) == {'message': 'Inspect file', 'fork_context': False}
    with pytest.raises(ValueError, match='Invalid arguments'):
        parse_output('</think><tool_call>' + json.dumps({'name': 'multi_agent_v1__spawn_agent',
            'arguments': {'message': 'Inspect file', 'fork_context': True}}) + '</tool_call>', routes)
    native = messages_from_wire([call, {'type': 'function_call_output',
        'call_id': call['call_id'], 'output': 'worker started'}])
    assert native[0]['tool_calls'][0]['function']['name'] == 'multi_agent_v1__spawn_agent'
    assert native[1]['tool_call_id'] == call['call_id']
    patch = '*** Begin Patch\n*** Add File: x\n+ok\n*** End Patch'
    result = parse_output('</think><tool_call>' + json.dumps(
        {'name': 'apply_patch', 'arguments': {'input': patch}}) + '</tool_call>', routes)
    assert result[-1]['type'] == 'custom_tool_call' and result[-1]['input'] == patch
    with pytest.raises(ValueError, match='Unknown'):
        parse_output('</think><tool_call><function=nonexistent></function></tool_call>', routes)
    with pytest.raises(ValueError, match='Incomplete'):
        parse_output('</think><tool_call><function=exec_command>', routes)
    with pytest.raises(ValueError, match='reasoning'):
        parse_output('not yet finished thinking', routes)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = ''.join(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n" for m in messages)
        if kwargs.get('add_generation_prompt'):
            text += '[assistant]<think>'
        return self.encode(text) if kwargs.get('tokenize', True) else text

    def encode(self, text, **kwargs):
        return list(text.encode())


@pytest.fixture
def info():
    return {'schema': 'remory.server.v1', 'identity': 'test-checkpoint', 'context_limit': 262144,
            'config': {'target_model': 'test-qwen', 'summary_contract': {
                'generation_suffix_ids': [999], 'max_summary_tokens': 8192}}}


def test_provider_compact_persists_then_restores_without_extra_model_calls(tmp_path, info):
    received = []
    handle = 'rm_' + '1' * 48
    def backend(request):
        body = json.loads(request.content)
        received.append((request.url.path, body))
        if request.url.path == '/v1/compact':
            assert body['history_ids'] and body['summary_ids']
            return httpx.Response(200, json={'handle': handle, 'receipt': {'soft_tokens': 64}})
        summary = body['input_ids'][-1] == 999
        return httpx.Response(200, json={'text': 'Remember file x.' if summary else
            'I recall it.</think>The file is x.', 'meta_info': {
                'finish_reason': {'type': 'stop'}, 'prompt_tokens': 100, 'completion_tokens': 10}})
    transport = httpx.MockTransport(backend)
    def make():
        return Provider('http://model', info, Tokenizer(), tmp_path,
                                 session_id='same-session', transport=transport)
    first = make()
    request = {'model': 'remory', 'instructions': 'Assist.', 'tools': [TOOL], 'input': [
        {'role': 'user', 'content': 'Remember file x.'}, {'type': 'compaction_trigger'}]}
    response = first.handle(request)
    assert response['output'][0]['type'] == 'compaction'
    assert len(received) == 2
    assert make().handle(request) == response
    assert len(received) == 2  # Retry replays the durable response, including its handle.
    continuation = {**request, 'input': response['output'] + [
        {'role': 'user', 'content': 'Which file?'}]}
    answer = make().handle(continuation)
    assert answer['output'][-1]['content'][0]['text'] == 'The file is x.'
    assert received[-1][1]['remory']['handle'] == handle
    assert b'Remember file x.' not in bytes(received[-1][1]['input_ids'])
    with pytest.raises(ValueError, match='changed'):
        make().handle({**continuation, 'instructions': 'Changed system prompt'})
    with pytest.raises(ValueError, match='checkpoint changed'):
        Provider('http://model', {**info, 'identity': 'new'}, Tokenizer(), tmp_path,
                 session_id='same-session', transport=transport)


def test_failed_compact_keeps_previous_state(tmp_path, info):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={
        'text': 'unfinished', 'meta_info': {'finish_reason': {'type': 'length'}}}))
    p = Provider('http://model', info, Tokenizer(), tmp_path, session_id='test', transport=transport)
    with pytest.raises(RuntimeError, match='original context'):
        p.handle({'input': [{'role': 'user', 'content': 'task'}, {'type': 'compaction_trigger'}]})
    assert not list(tmp_path.glob('thread-*.json'))


def test_provider_http_auth_and_responses_stream(tmp_path, info):
    from fastapi.testclient import TestClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={
        'text': '</think>Hello.', 'meta_info': {'finish_reason': {'type': 'stop'}}}))
    p = Provider('http://model', info, Tokenizer(), tmp_path, session_id='test', transport=transport)
    client = TestClient(create_app(p, token='local'))
    assert client.get('/health').status_code == 401
    response = client.post('/v1/responses', headers={'Authorization': 'Bearer local'},
        json={'stream': True, 'input': [{'role': 'user', 'content': 'Hello'}]})
    assert response.status_code == 200
    assert 'event: response.completed' in response.text and 'Hello.' in response.text
    assert response.headers['content-type'].startswith('text/event-stream')


def test_completed_child_is_folded_once_and_survives_provider_restart(tmp_path, info):
    requests = []
    serial = 0
    def backend(request):
        nonlocal serial
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == '/v1/compact':
            serial += 1
            return httpx.Response(200, json={'handle': 'rm_' + f'{serial:048x}',
                                            'receipt': {'soft_tokens': 64}})
        return httpx.Response(200, json={'text': 'Summary.' if body['input_ids'][-1] == 999
            else '</think>Task complete.', 'meta_info': {'finish_reason': {'type': 'stop'}}})
    def make():
        return Provider('http://model', info, Tokenizer(), tmp_path, session_id='test',
                        transport=httpx.MockTransport(backend))
    root_headers = {'thread-id': 'parent', 'x-codex-turn-metadata': json.dumps({'thread_id': 'parent'})}
    child_headers = {'thread-id': 'child', 'x-codex-turn-metadata': json.dumps({
        'thread_id': 'child', 'parent_thread_id': 'parent'})}
    root = {'instructions': 'Assist.', 'input': [{'role': 'user', 'content': 'Parent task'}]}
    p = make()
    answer = p.handle(root, root_headers)
    p.handle({'instructions': 'Assist.', 'input': [{'role': 'user', 'content': 'Child task'}]}, child_headers)
    new_input = root['input'] + answer['output'] + [{'role': 'user', 'content': 'Continue parent'}]
    answer = make().handle({**root, 'input': new_input}, root_headers)
    folds = [body for path, body in requests if path == '/v1/compact' and body.get('branches')]
    assert len(folds) == 1 and len(folds[0]['branches']) == 1
    next_input = new_input + answer['output'] + [{'role': 'user', 'content': 'Continue again'}]
    make().handle({**root, 'input': next_input}, root_headers)
    assert len([b for path, b in requests if path == '/v1/compact' and b.get('branches')]) == 1
    assert requests[-1][1]['remory']['handle'] == 'rm_' + f'{2:048x}'


def test_zero_argument_tools_and_single_assistant_turn():
    _, routes = tools_from_wire([{'type': 'function', 'name': 'get_goal', 'parameters': {
        'type': 'object', 'properties': {}, 'additionalProperties': False}}])
    output = parse_output('Check progress.</think>Inspecting the goal.'
        '<tool_call><function=get_goal></function></tool_call>', routes)
    call = output[-1]
    assert json.loads(call['arguments']) == {}
    messages = messages_from_wire(output + [{'type': 'function_call_output',
        'call_id': call['call_id'], 'output': 'Active'}])
    assert len(messages) == 2
    assert messages[0]['reasoning_content'] == 'Check progress.'
    assert messages[0]['content'] == 'Inspecting the goal.'
    assert len(messages[0]['tool_calls']) == 1
    with pytest.raises(ValueError, match='Missing tool outputs'):
        messages_from_wire(output)
    with pytest.raises(ValueError, match='no pending call'):
        messages_from_wire([{'type': 'function_call_output', 'call_id': 'unknown', 'output': 'ok'}])


def test_quoted_vision_markers_are_text():
    codec = CodexChatTemplate(Tokenizer(), tools=[])
    prefix, history = codec.split([{'role': 'system', 'content': 'Assist.'}], [
        {'role': 'user', 'content': 'Explain <|image_pad|> and <|vision_start|>.'}])
    text = bytes(prefix + history).decode()
    assert '<|image_pad|>' not in text and r'\u003c|image_pad|\u003e' in text
    assert '<|vision_start|>' not in text
    assert '<|im_start|>' in text  # Native chat delimiters are preserved.


def test_interrupted_undelivered_tool_is_not_replayed_on_new_turn(tmp_path, info):
    received = []
    def backend(request):
        body = json.loads(request.content)
        received.append(body)
        text = ('</think><tool_call><function=exec_command><parameter=cmd>pwd</parameter>'
                '</function></tool_call>') if len(received) == 1 else '</think>Cancelled.'
        return httpx.Response(200, json={'text': text, 'meta_info': {'finish_reason': {'type': 'stop'}}})
    def make():
        return Provider('http://model', info, Tokenizer(), tmp_path, session_id='test',
                        transport=httpx.MockTransport(backend))
    first = {'tools': [TOOL], 'input': [{'role': 'user', 'content': 'Inspect project'}]}
    proposal = make().handle(first)  # Generation completed, but the UI did not receive the tool.
    assert proposal['output'][-1]['type'] == 'function_call'
    next_turn = {**first, 'input': first['input'] + [{'role': 'user', 'content':
        '<turn_aborted>Interrupted</turn_aborted>'}, {'role': 'user', 'content': 'Cancel that'}]}
    answer = make().handle(next_turn)
    assert answer['output'][-1]['content'][0]['text'] == 'Cancelled.'
    assert b'pwd' not in bytes(received[-1]['input_ids'])


def test_final_answer_can_be_compacted_then_resumed(tmp_path, info):
    calls = []
    def backend(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == '/v1/compact':
            return httpx.Response(200, json={'handle': 'rm_' + '2' * 48, 'receipt': {}})
        text = 'Completed work.' if body['input_ids'][-1] == 999 else '</think>Done.'
        return httpx.Response(200, json={'text': text, 'meta_info': {'finish_reason': {'type': 'stop'}}})
    p = Provider('http://model', info, Tokenizer(), tmp_path, session_id='test',
                 transport=httpx.MockTransport(backend))
    history = [{'role': 'user', 'content': 'Finish task'}]
    final = p.handle({'input': history})
    checkpoint = p.handle({'input': history + final['output'] + [{'type': 'compaction_trigger'}]})
    assert checkpoint['output'][0]['type'] == 'compaction'
    answer = p.handle({'input': checkpoint['output'] + [{'role': 'user', 'content': 'Next task'}]})
    assert answer['output'][-1]['content'][0]['text'] == 'Done.'
    assert calls[-1][1]['remory']['handle'] == 'rm_' + '2' * 48


def test_provider_ignores_ambient_http_proxy(tmp_path, info, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"text":"Direct local connection"}')

        def log_message(self, *args):
            pass

    monkeypatch.setenv('ALL_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '')
    monkeypatch.setenv('no_proxy', '')
    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            p = Provider(f'http://127.0.0.1:{server.server_port}', info, Tokenizer(), tmp_path,
                         session_id='test')
            with p.client() as client:
                assert client.generate(input_ids=[1])['text'] == 'Direct local connection'
        finally:
            server.shutdown()
            thread.join(timeout=5)
