import json
import threading
from types import SimpleNamespace

import pytest

from remory.codex.native_folding import Director


def tool(name, properties, required):
    return {'type': 'function', 'name': name, 'parameters': {
        'type': 'object', 'properties': properties, 'required': required,
        'additionalProperties': False}}


TOOLS = [{'type': 'namespace', 'name': 'multi_agent_v1', 'tools': [
    tool('spawn_agent', {'message': {'type': 'string'}, 'fork_context': {'type': 'boolean'}},
         ['message']),
    tool('wait', {'ids': {'type': 'array', 'items': {'type': 'string'}},
                  'timeout_ms': {'type': 'integer'}}, ['ids']),
    tool('close_agent', {'id': {'type': 'string'}}, ['id'])]}]
HEADERS = {'thread-id': 'root'}


def test_director_emits_real_native_calls_and_holds_close_until_commit_context_exit():
    state = SimpleNamespace(result=None)
    d = Director(None, state)
    request = {'tools': TOOLS, 'input': []}
    spawn = d.handle(request, HEADERS)
    call = spawn['output'][0]
    assert call['namespace'] == 'multi_agent_v1' and call['name'] == 'spawn_agent'
    assert json.loads(call['arguments'])['fork_context'] is False
    assert d.handle(request, HEADERS) == spawn
    request['input'] = [call, {'type': 'function_call_output', 'call_id': call['call_id'],
                               'output': json.dumps({'agent_id': 'child'})}]
    wait = d.handle(request, HEADERS)['output'][0]
    assert wait['name'] == 'wait'
    state.result = {'messages': [], 'summary': 'checkpoint'}
    request['input'].append({'type': 'function_call_output', 'call_id': wait['call_id'],
                             'output': '{}'})
    response = []
    thread = threading.Thread(target=lambda: response.append(d.handle(request, HEADERS)))
    thread.start()
    assert d.ready.wait(2)
    assert not response
    d.release.set()
    thread.join(2)
    close = response[0]['output'][0]
    assert close['name'] == 'close_agent'
    request['input'].append({'type': 'function_call_output', 'call_id': close['call_id'],
                             'output': '{"previous_status":"completed"}'})
    d.handle(request, HEADERS)
    assert d.closed.is_set()


def test_close_error_never_counts_as_cleanup():
    d = Director(None, SimpleNamespace(result={}))
    d.root, d.phase = 'root', 'closed'
    d.calls = [{'call_id': 'close'}]
    with pytest.raises(RuntimeError, match='acknowledge'):
        d.handle({'input': [{'type': 'function_call_output', 'call_id': 'close',
                            'output': '{"error":"denied"}'}]}, HEADERS)
    assert not d.closed.is_set()


def test_commit_requires_later_model_turn_and_unknown_tools_rejected():
    state = SimpleNamespace(test_transform=lambda **kw: {'candidate_id': 'a'},
                            commit_transform=lambda **kw: {'committed': True})
    d = Director(None, state)
    d.child_turn = 1
    assert d.tool('test_transform', {'edits': [], 'summary': 's'})['candidate_id'] == 'a'
    assert 'error' in d.tool('commit_transform', {'candidate_id': 'a'})
    d.child_turn = 2
    assert d.tool('commit_transform', {'candidate_id': 'a'})['committed']
    with pytest.raises(ValueError, match='Unknown'):
        d.tool('exec_command', {})


def test_no_recursive_compact_or_foreign_child():
    d = Director(None, SimpleNamespace(result=None))
    with pytest.raises(ValueError, match='Recursive'):
        d.handle({'input': [{'type': 'compaction_trigger'}]}, HEADERS)
    with pytest.raises(ValueError, match='Only the native'):
        d.handle({'input': []}, {'thread-id': 'foreign'})


@pytest.mark.parametrize(('parent_failure', 'reported_tokens'),
                         [(False, 10), (True, 10), ('close', 10), (False, 190000),
                          ('long-fold', 10), ('long-page', 10), ('native-estimate', 270000),
                          ('actor-budget', 10), ('quoted-history', 10)])
def test_native_binary_fake_model_smoke(tmp_path, monkeypatch, parent_failure, reported_tokens):
    """Opt-in CPU integration: real Codex spawns child; model transport is fake."""
    import os
    import re
    import time
    from pathlib import Path
    import httpx
    from remory.codex.cli import build_catalog
    from remory.codex.native_folding import NativeFolder
    from remory.codex.provider import Provider
    from test_codex import Tokenizer
    binary = os.environ.get('REMORY_TEST_CODEX')
    if not binary:
        pytest.skip('Set REMORY_TEST_CODEX to the pinned binary for CPU native smoke')
    clock_offset = [0]
    if parent_failure == 'long-fold':
        # Advance only the folder's clock while a real native child is decoding.
        # Transport/thread clocks stay real; no hour-long test or GPU is needed.
        monkeypatch.setattr('remory.codex.native_folding.time', SimpleNamespace(
            monotonic=lambda: time.monotonic() + clock_offset[0],
            sleep=time.sleep, time=time.time))
    (tmp_path / '.codex').mkdir()
    catalog = build_catalog(Path(binary), tmp_path, os.environ.copy(), 262144, 'Fake local')
    original_catalog = catalog.read_bytes()
    turns = []
    long_page = '页首证据' + '保留中文与边界' * 700 + '页中证据' + '保持科学限制' * 700 + '页尾证据'
    class RestrictedTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            names = [t['function']['name'] for t in kwargs.get('tools', [])]
            assert len(names) == 4 and all(n.startswith('mcp__fold__') for n in names)
            return super().apply_chat_template(messages, **kwargs)

    def backend(request):
        assert request.url.path == '/generate'
        body = json.loads(request.content)
        assert body['sampling_params']['max_new_tokens'] == (81920 if parent_failure == 'actor-budget' else 32768)
        text = bytes(body['input_ids']).decode().replace('\\"', '"')
        turns.append(text)
        if len(turns) == 1:
            if parent_failure == 'long-fold':
                clock_offset[0] = 3601
                time.sleep(.3)
            name, args = 'inspect_context', {'target': 'messages', 'limit': 12000}
        elif len(turns) == 2:
            if parent_failure == 'long-page':
                assert long_page in text, 'Native transport truncated the inspected source page'
            name, args = 'test_transform', {'edits': [], 'summary': 'Preserve task and constraints.'}
        elif len(turns) == 3:
            candidates = re.findall(r'"candidate_id"\s*:\s*"([^"]+)"', text)
            assert candidates, text[-2000:]
            name, args = 'commit_transform', {'candidate_id': candidates[-1]}
        else:
            return httpx.Response(200, json={'text': '</think>Organizer complete.',
                'meta_info': {'finish_reason': {'type': 'stop'}, 'prompt_tokens': reported_tokens, 'completion_tokens': 5}})
        reasoning = (r'Copied message: \n</think>\n<tool_call> exec ...' + '\n'
                     if parent_failure == 'quoted-history' else '')
        return httpx.Response(200, json={'text': reasoning + '</think><tool_call>' + json.dumps({
            'name': 'mcp__fold__' + name, 'arguments': args}) + '</tool_call>',
            'meta_info': {'finish_reason': {'type': 'stop'}, 'prompt_tokens': reported_tokens, 'completion_tokens': 5}})

    info = {'identity': 'fake', 'config': {'summary_contract': {}}}
    provider = Provider('http://fake', info, RestrictedTokenizer(), tmp_path / 'provider',
                        session_id='parent', transport=httpx.MockTransport(backend))
    if parent_failure == 'actor-budget':
        provider.organizer_max_new_tokens = 81920
    runtime = {'model': 'remory', 'model_catalog_json': str(catalog),
               'model_context_window': 262144, 'model_auto_compact_token_limit': 180000,
               'features': {'multi_agent': True}, 'agents': {'max_threads': 2, 'max_depth': 1},
               'model_reasoning_effort': 'high', 'web_search': 'disabled'}
    snapshot = {'messages': [{'role': 'user', 'content': 'Keep this task.'},
                            {'role': 'assistant', 'content': long_page if parent_failure == 'long-page' else 'Progress.'}],
                'prefix': [], 'previous_summary': None, 'branch_summaries': []}
    result = None
    folder = NativeFolder(binary, runtime, os.environ.copy(), provider)
    try:
        with folder.fold(snapshot) as result:
            assert result['messages'] == snapshot['messages']
            assert result['audit']['transport'] == 'native-codex-spawn'
            assert result['audit']['director_thread'] != result['audit']['organizer_thread']
            assert result['audit']['reasoning_effort'] == 'high'
            assert result['usage'] == {'input_tokens': 4 * reported_tokens, 'output_tokens': 20,
                                       'total_tokens': 4 * reported_tokens + 20}
            if parent_failure == 'close':
                folder.close()
                assert folder._cancelled.is_set()
                assert all(process.poll() is not None for process in folder._active.values() if process)
            elif parent_failure is True:
                raise ValueError('parent encode failed')
    except ValueError as exc:
        assert parent_failure and str(exc) == 'parent encode failed'
    assert result['audit']['child_closed']
    assert result['audit']['cleanup_issues'] == []
    assert not list(provider.directory.glob('fold-cleanup-*.json'))
    assert len(turns) == 4
    assert catalog.read_bytes() == original_catalog
    lifecycle = json.loads(next(provider.directory.glob('fold-lifecycle-*.json')).read_text())
    assert lifecycle['context_limit'] == 262144  # Real capacity, not native estimates.


def test_native_tool_results_accept_text_blocks():
    from remory.codex.native_folding import tool_result_object
    assert tool_result_object([{'type': 'text', 'text': '{"agent_id":"child"}'}]) == {
        'agent_id': 'child'}
    with pytest.raises(ValueError):
        tool_result_object('[]')


def test_inspection_transport_preserves_unicode_and_untrusted_json_text():
    from copy import deepcopy
    from remory.codex.native_folding import fold_tool_content, unwrap_fold_outputs
    page = '{"content":"古文证据与失败记录\\n literal \\u4e2d"}'
    content = fold_tool_content({'target': 'messages', 'offset': 0, 'text': page, 'next_offset': None})
    assert content[1]['text'] == page
    assert '\\u53e4' not in ''.join(x['text'] for x in content)
    output = json.dumps({'content': content, 'isError': False})
    request = {'input': [
        {'type': 'message', 'role': 'user', 'content': output},
        {'type': 'function_call_output', 'call_id': 'foreign', 'output': output},
        {'type': 'function_call_output', 'call_id': 'owned', 'output': output},
        {'type': 'function_call_output', 'call_id': 'native',
         'output': 'Wall time: 0.0211 seconds\nOutput:\n' + json.dumps(content)}]}
    before = deepcopy(request)
    unwrap_fold_outputs(request, {'owned', 'native'})
    assert request['input'][:2] == before['input'][:2]
    assert request['input'][2]['output'].endswith('\n' + page)
    assert 'literal \\u4e2d' in request['input'][2]['output']
    assert request['input'][3]['output'] == request['input'][2]['output']


@pytest.mark.parametrize(('inputs', 'capacity', 'reserve', 'expected', 'finish'), [
    (4, None, None, 2, 'stop'), (4, 20, 8, 8, 'stop'),
    (233932, 262138, 32768, 28206, 'stop'),
    (233932, 262138, 32768, 28206, 'length'),
    (206500, 262138, 81920, 55638, 'stop'),
    (206500, 262138, 81920, 55638, 'length'),
    (4, 5, 8, 1, 'length'), (4, 4, 8, 0, 'stop'), (5, 4, 8, 0, 'stop'),
])
def test_organizer_checks_actual_capacity_without_truncating(
        monkeypatch, inputs, capacity, reserve, expected, finish):
    from contextlib import nullcontext
    from remory.codex.native_folding import TOOLS as FOLD_TOOLS, REQUIRED
    from remory.codex.provider import Provider
    prefix, tail = [1, 2], [3] * (inputs - 2)
    codec = SimpleNamespace(split=lambda *args, **kwargs: (prefix, tail))
    monkeypatch.setattr(Provider, '_context', lambda *args: (codec, [], [], None, {}, False, []))
    generated_calls = []
    def generate(**kwargs):
        generated_calls.append(kwargs)
        assert kwargs['input_ids'] == prefix + tail
        assert kwargs['max_new_tokens'] == expected
        if capacity:
            assert len(kwargs['input_ids']) + kwargs['max_new_tokens'] <= capacity
        return {'text': '</think>Ready.', 'meta_info': {
            'finish_reason': {'type': finish}, 'prompt_tokens': inputs, 'completion_tokens': 1}}
    provider = SimpleNamespace(max_new_tokens=2,
        client=lambda: nullcontext(SimpleNamespace(generate=generate)))
    d = Director(provider, SimpleNamespace(result=None),
                 context_limit=capacity, max_new_tokens=reserve)
    d.root = 'root'
    request = {'input': [], 'tools': [{'type': 'namespace', 'name': 'mcp__fold',
        'tools': [tool(name, props, REQUIRED[name]) for name, props in FOLD_TOOLS.items()]}]}
    headers = {'thread-id': 'child', 'x-codex-turn-metadata': '{"parent_thread_id":"root"}'}
    if not expected:
        with pytest.raises(ValueError, match=f'{inputs} input tokens and has no room for output'):
            d.handle(request, headers)
        assert not generated_calls and not d.child_turn
    elif finish != 'stop':
        with pytest.raises(RuntimeError, match=f'finish_reason=length, .*output_limit={expected}'):
            d.handle(request, headers)
    else:
        assert d.handle(request, headers)['status'] == 'completed'
    assert d.max_input_tokens == inputs
    assert d.state.result is None
    assert request['input'] == []


def test_preclosed_folder_never_spawns(monkeypatch):
    from remory.codex.native_folding import NativeFolder
    def forbidden(*args, **kwargs):
        pytest.fail('Closed folder attempted to spawn a process')
    monkeypatch.setattr('subprocess.Popen', forbidden)
    folder = NativeFolder('unused', {}, {}, None)
    folder.close()
    folder.close()
    with pytest.raises(RuntimeError, match='closed'):
        with folder.fold({}):
            pytest.fail('Closed folder yielded')
    assert folder._active == {}
