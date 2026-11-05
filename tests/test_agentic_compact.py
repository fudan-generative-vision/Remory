from contextlib import contextmanager
from copy import deepcopy
import json

import httpx
import pytest

from remory.codex.folding import FoldState
from remory.codex.provider import Provider
from remory.codex.terminal import CommandAliases
from test_codex import Tokenizer


def snapshot():
    return {'messages': [
        {'role': 'user', 'content': 'Keep the actual measurements and unfinished work.'},
        {'role': 'assistant', 'content': 'Checking.', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'exec', 'arguments': {'cmd': 'test'}}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'exit 1: test failed\n' + 'details ' * 100},
        {'role': 'assistant', 'content': 'Need to fix test. ' * 100}],
        'previous_summary': 'Previous task', 'prefix': [{'role': 'system', 'content': 'Immutable'}]}


def test_fold_copy_protection_pairs_and_provenance():
    source = snapshot()
    before = deepcopy(source)
    state = FoldState(source)
    source['messages'][0]['content'] = 'outside mutation'
    assert 'actual measurements' in state.inspect_context('messages')['text']
    with pytest.raises(ValueError, match='protected'):
        state.test_transform([{'start': 0, 'end': 1, 'text': 'lost task'}], 'Summary')
    with pytest.raises(ValueError, match='split a tool'):
        state.test_transform([{'start': 1, 'end': 2, 'text': 'lost result'}], 'Summary')
    edits = [{'start': 1, 'end': 3, 'text': 'Ran test; exit 1, failed. Fix is still pending.'}]
    trial = state.test_transform(edits, 'Test failed. Fix remains pending.')
    assert trial['folded_characters'] < trial['original_characters']
    assert state.result is None
    state.commit_transform(trial['candidate_id'])
    result = state.result
    assert result['messages'][0] == before['messages'][0]
    assert result['messages'][-1] == before['messages'][-1]
    assert result['audit']['edits'] == edits
    result['messages'][0]['content'] = 'mutate returned copy'
    assert state.result['messages'][0] == before['messages'][0]


def test_fold_rejects_incomplete_data_and_untested_commit():
    source = snapshot()
    source['messages'].pop(2)
    with pytest.raises(ValueError, match='Tool results'):
        FoldState(source)
    state = FoldState(snapshot())
    with pytest.raises(ValueError, match='before committing'):
        state.commit_transform('not-tested')
    state.abort_folding('Evidence is ambiguous')
    assert state.result is None
    with pytest.raises(ValueError, match='aborted'):
        state.test_transform([], 'No changes')


def test_summary_contract_is_checked_before_candidate_commit():
    source = snapshot()
    source['summary_contract'] = {'output_schema': {'type': 'object',
        'properties': {'task': {'type': 'string'}}, 'required': ['task']}}
    state = FoldState(source)
    with pytest.raises(ValueError, match='output_schema'):
        state.test_transform([], 'Not the required JSON')
    trial = state.test_transform([], '{"task":"Keep failures"}')
    assert trial['valid']


class Folder:
    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail

    @contextmanager
    def fold(self, snapshot):
        self.events.append('spawn')
        try:
            if self.fail:
                raise RuntimeError('injected organizer failure')
            state = FoldState(snapshot)
            trial = state.test_transform([{'start': 1, 'end': 2, 'text': 'Observed test failure.'}],
                                         'Test failed; fix remains pending.')
            state.commit_transform(trial['candidate_id'])
            yield state.result
        finally:
            self.events.append('close')


def provider(tmp_path, events):
    def backend(request):
        if request.method == 'DELETE':
            events.append('delete')
            return httpx.Response(200, json={'deleted': True})
        assert request.url.path == '/v1/compact', 'agentic path must not run ordinary summary'
        body = json.loads(request.content)
        events.append(('encode', body))
        return httpx.Response(200, json={'handle': 'rm_' + '1' * 48, 'receipt': {'soft_tokens': 64}})
    p = Provider('http://local', {'identity': 'test', 'config': {'target_model': 'local-qwen'}},
                 Tokenizer(), tmp_path, session_id='session', transport=httpx.MockTransport(backend))
    p.folder = Folder(events)
    return p


def compact_request(trigger=True):
    return {'instructions': 'Assist.', 'input': [
        {'role': 'user', 'content': 'Keep failures.'},
        {'role': 'assistant', 'content': 'Detailed observations ' * 100},
    ] + ([{'type': 'compaction_trigger'}] if trigger else [])}


@pytest.mark.parametrize('force', [False, True])
def test_manual_and_automatic_compact_fold_before_encode_and_close_after_commit(tmp_path, force):
    events = []
    p = provider(tmp_path, events)
    p.agentic_compact_command()
    request = compact_request(not force)
    response = p.handle(request, force_compact=force)
    assert response['output'][0]['type'] == 'compaction'
    assert events[0] == 'spawn' and events[-1] == 'close'
    body = events[1][1]
    assert b'Observed test failure.' in bytes(body['history_ids'])
    assert b'Detailed observations' not in bytes(body['history_ids'])
    assert p._load('root')['checkpoints'][response['output'][0]['encrypted_content']]['folding']['edits']
    restored = provider(tmp_path, events)
    assert restored.meta['agentic_compact'] is True
    assert restored.handle(request, force_compact=force) == response
    assert len(events) == 3


def test_failed_fold_or_checkpoint_write_keeps_parent_and_cleans_new_memory(tmp_path, monkeypatch):
    events = []
    p = provider(tmp_path, events)
    p.agentic_compact_command()
    p.folder.fail = True
    with pytest.raises(RuntimeError, match='organizer failure'):
        p.handle(compact_request())
    assert not p._path('root').exists()
    assert events == ['spawn', 'close']
    p.folder.fail = False
    import remory.codex.provider as module
    save = module.atomic_json
    def fail_thread(path, value):
        if path == p._path('root'):
            raise OSError('disk full')
        return save(path, value)
    monkeypatch.setattr(module, 'atomic_json', fail_thread)
    with pytest.raises(OSError, match='disk full'):
        p.handle(compact_request())
    assert events[-2:] == ['delete', 'close']
    assert not p._path('root').exists()


def test_session_command_does_not_infer_enable_from_history_or_call_model(tmp_path):
    events = []
    p = provider(tmp_path, events)
    p.session_commands = True
    p.handle({'input': '[Remory session] /agentic_compact'})
    assert p.meta['agentic_compact']
    assert not events
    assert p._control({'input': [{'role': 'user', 'content': '[Remory session] /agentic_compact'},
        {'role': 'assistant', 'content': 'ack'}]}, 'root') is None
    assert p._control({'input': '[Remory session] /agentic_compact'}, 'child') is None
    p.handle({'input': '[Remory session] /agentic_compact off'})
    assert not p.meta['agentic_compact']
    p.handle({'input': '[Remory session] /agentic_compact status'})
    assert not p.meta['agentic_compact']


@pytest.mark.parametrize('input_bytes,expected', [
    (b'/agentic_compact\r', b'\x1b[200~[Remory session] /agentic_compact\x1b[201~\r'),
    (b'/agentic_compact off\r', b'\x1b[200~[Remory session] /agentic_compact off\x1b[201~\r'),
    (b'/compact\r', b'/compact\r'),
    (b'quote /agentic_compact\r', b'quote /agentic_compact\r'),
    (b'/agentic_compact_extra\r', b'/agentic_compact_extra\r'),
    (b'\x1b[200~/agentic_compact\x1b[201~\r', b'\x1b[200~[Remory session] /agentic_compact\x1b[201~\r'),
    (b'\x1b[A/agentic_compact\r', b'\x1b[A/agentic_compact\r'),
    (b'\x1b[?1u\x1b[2;1R/agentic_compact\x1b[13u',
     b'\x1b[?1u\x1b[2;1R\x1b[200~[Remory session] /agentic_compact\x1b[201~\x1b[13u'),
])
def test_terminal_alias_chunking_paste_and_ordinary_input(input_bytes, expected):
    for chunk_size in range(1, len(input_bytes) + 1):
        aliases = CommandAliases()
        actual = b''.join(aliases.feed(input_bytes[i:i + chunk_size])
                          for i in range(0, len(input_bytes), chunk_size))
        assert actual == expected


def test_terminal_escape_is_not_swallowed():
    aliases = CommandAliases()
    assert aliases.feed(b'\x1b') == b''
    assert aliases.flush_escape() == b'\x1b'
