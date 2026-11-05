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


@pytest.mark.parametrize('parent_failure', [False, True, 'close'])
def test_native_binary_fake_model_smoke(tmp_path, parent_failure):
    """Opt-in CPU integration: real Codex spawns child; model transport is fake."""
    import os
    import re
    from pathlib import Path
    import httpx
    from remory.codex.cli import build_catalog
    from remory.codex.native_folding import NativeFolder
    from remory.codex.provider import Provider
    from test_codex import Tokenizer
    binary = os.environ.get('REMORY_TEST_CODEX')
    if not binary:
        pytest.skip('Set REMORY_TEST_CODEX to the pinned binary for CPU native smoke')
    (tmp_path / '.codex').mkdir()
    catalog = build_catalog(Path(binary), tmp_path, os.environ.copy(), 262144, 'Fake local')
    turns = []
    class RestrictedTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            names = [t['function']['name'] for t in kwargs.get('tools', [])]
            assert len(names) == 4 and all(n.startswith('mcp__fold__') for n in names)
            return super().apply_chat_template(messages, **kwargs)

    def backend(request):
        assert request.url.path == '/generate'
        body = json.loads(request.content)
        text = bytes(body['input_ids']).decode().replace('\\"', '"')
        turns.append(text)
        if len(turns) == 1:
            name, args = 'inspect_context', {'target': 'messages'}
        elif len(turns) == 2:
            name, args = 'test_transform', {'edits': [], 'summary': 'Preserve task and constraints.'}
        elif len(turns) == 3:
            candidates = re.findall(r'"candidate_id"\s*:\s*"([^"]+)"', text)
            assert candidates, text[-2000:]
            name, args = 'commit_transform', {'candidate_id': candidates[-1]}
        else:
            return httpx.Response(200, json={'text': '</think>Organizer complete.',
                'meta_info': {'finish_reason': {'type': 'stop'}, 'prompt_tokens': 10, 'completion_tokens': 5}})
        return httpx.Response(200, json={'text': '</think><tool_call>' + json.dumps({
            'name': 'mcp__fold__' + name, 'arguments': args}) + '</tool_call>',
            'meta_info': {'finish_reason': {'type': 'stop'}, 'prompt_tokens': 10, 'completion_tokens': 5}})

    info = {'identity': 'fake', 'config': {'summary_contract': {}}}
    provider = Provider('http://fake', info, RestrictedTokenizer(), tmp_path / 'provider',
                        session_id='parent', transport=httpx.MockTransport(backend))
    runtime = {'model': 'remory', 'model_catalog_json': str(catalog),
               'features': {'multi_agent': True}, 'agents': {'max_threads': 2, 'max_depth': 1},
               'model_reasoning_effort': 'high', 'web_search': 'disabled'}
    snapshot = {'messages': [{'role': 'user', 'content': 'Keep this task.'},
                            {'role': 'assistant', 'content': 'Progress.'}],
                'prefix': [], 'previous_summary': None, 'branch_summaries': []}
    result = None
    folder = NativeFolder(binary, runtime, os.environ.copy(), provider)
    try:
        with folder.fold(snapshot) as result:
            assert result['messages'] == snapshot['messages']
            assert result['audit']['transport'] == 'native-codex-spawn'
            assert result['audit']['director_thread'] != result['audit']['organizer_thread']
            assert result['audit']['reasoning_effort'] == 'high'
            assert result['usage'] == {'input_tokens': 40, 'output_tokens': 20, 'total_tokens': 60}
            if parent_failure == 'close':
                folder.close()
                assert folder._cancelled.is_set()
                assert all(process.poll() is not None for process in folder._active.values() if process)
            elif parent_failure:
                raise ValueError('parent encode failed')
    except ValueError as exc:
        assert parent_failure and str(exc) == 'parent encode failed'
    assert result['audit']['child_closed']
    assert result['audit']['cleanup_issues'] == []
    assert not list(provider.directory.glob('fold-cleanup-*.json'))
    assert len(turns) == 4


def test_native_tool_results_accept_text_blocks():
    from remory.codex.native_folding import tool_result_object
    assert tool_result_object([{'type': 'text', 'text': '{"agent_id":"child"}'}]) == {
        'agent_id': 'child'}
    with pytest.raises(ValueError):
        tool_result_object('[]')


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
