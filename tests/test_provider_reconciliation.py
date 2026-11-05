"""Durable child response retries repair routing without repeating model work."""
import json

import httpx
import pytest

import remory.codex.provider as module
from test_codex import Tokenizer


def test_child_routing_recovers_after_metadata_write_failure(tmp_path, monkeypatch):
    calls = []
    info = {'identity': 'fixed', 'config': {'target_model': 'fake',
        'summary_contract': {'generation_suffix_ids': [999]}}}
    def backend(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == '/v1/compact':
            return httpx.Response(200, json={'handle': 'rm_' + '1' * 48, 'receipt': {}})
        return httpx.Response(200, json={'text': '</think>Complete.',
            'meta_info': {'finish_reason': {'type': 'stop'}}})
    def make():
        return module.Provider('http://fake', info, Tokenizer(), tmp_path,
            session_id='session', transport=httpx.MockTransport(backend))
    provider = make()
    root_request = {'input': [{'role': 'user', 'content': 'Root task'}]}
    root_answer = provider.handle(root_request, {'thread-id': 'root'})
    headers = {'thread-id': 'child', 'x-codex-turn-metadata': json.dumps(
        {'thread_id': 'child', 'parent_thread_id': 'root'})}
    request = {'input': [{'role': 'user', 'content': 'Child task'}]}
    original = module.atomic_json
    def fail_completion(path, value):
        if path == provider.meta_path and value.get('threads', {}).get('child', {}).get('completion'):
            raise OSError('injected metadata disk failure')
        return original(path, value)
    monkeypatch.setattr(module, 'atomic_json', fail_completion)
    with pytest.raises(OSError, match='metadata disk failure'):
        provider.handle(request, headers)
    state_bytes = provider._path('child').read_bytes()
    state = provider._load('child')
    assert state['completion'] and state['responses']
    assert 'completion' not in json.loads(provider.meta_path.read_text())['threads']['child']
    assert 'completion' not in provider.meta['threads']['child']
    count = len(calls)
    # A second failed metadata save must also leave the durable memory untouched.
    restored = make()
    with pytest.raises(OSError, match='metadata disk failure'):
        restored.handle(request, headers)
    assert len(calls) == count and provider._path('child').read_bytes() == state_bytes
    monkeypatch.setattr(module, 'atomic_json', original)
    restored = make()
    response = restored.handle(request, headers)
    assert response == next(iter(state['responses'].values()))
    assert len(calls) == count and provider._path('child').read_bytes() == state_bytes
    assert make().meta['threads']['child']['completion'] == state['completion']
    meta_bytes = provider.meta_path.read_bytes()
    assert restored.handle(request, headers) == response
    assert provider.meta_path.read_bytes() == meta_bytes and len(calls) == count
    # The repaired descriptor participates in the existing parent branch route.
    make().handle({'input': root_request['input'] + root_answer['output'] +
        [{'role': 'user', 'content': 'Continue'}]}, {'thread-id': 'root'})
    folds = [body for path, body in calls if path == '/v1/compact' and body.get('branches')]
    assert len(folds) == 1 and folds[0]['branches'] == [state['completion']['handle']]
