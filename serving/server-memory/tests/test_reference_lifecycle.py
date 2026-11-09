import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_remory_server_memory import engine, client, build, publish
from context_compression.remory_memory import RemoteMemory
from sglang.srt.memory_handle_registry import TPMemoryHandleRegistry
from sglang.srt.remory_memory import memory_operation, collective_publish


@pytest.mark.parametrize('budget,expected', [('default', 4096), (4096, 4096), (4480, 4480), (None, None)])
def test_remote_budget_preserves_request_policy(engine, budget, expected):
    remote = client(engine)
    kwargs = {} if budget == 'default' else dict(memory_budget=budget)
    result = build(remote, [3] * (70 * 1024), **kwargs)
    assert result['receipt']['memory_budget'] == expected
    value = result['memory_embeddings']
    assert isinstance(value, RemoteMemory)
    assert len(value) == value.shape[0]
    if expected == 4096:
        assert value.shape[0] <= 4096 and result['receipt']['parent_count'] > 0
    else:
        assert value.shape[0] == 4480 and result['receipt']['parent_count'] == 0
    assert engine.registry.stats()['entries'] == 1
    wire = json.dumps(engine.requests)
    assert 'summary_features' not in wire and 'memory_embeddings' not in wire
    remote.release_memory(value)


def test_changed_request_budget_is_not_cached(engine):
    c = client(engine)
    for budget in (4096, 4480, 4096):
        c._residual_config['residual']['memory_budget'] = budget
        result = build(c, [3] * (70 * 1024))
        assert result['receipt']['memory_budget'] == budget
        c.release_memory(result['memory_embeddings'])


def fresh_registry(*, clock=None, entries=2, nbytes=4096):
    return TPMemoryHandleRegistry(server_epoch='test-epoch', tp_rank=0,
        ttl_seconds=10, max_entries=entries, max_bytes=nbytes,
        hook_abi_digest='sha256:' + 'a' * 64, require_cuda=False,
        **({} if clock is None else dict(monotonic=lambda: clock[0])))


def test_expiry_renewal_and_retained_forward_reference():
    clock = [0.]
    r = fresh_registry(clock=clock)
    handle = publish(r, torch.ones(64, 2))
    admitted = r.resolve(handle).tensor
    clock[0] = 9
    r.renew(handle)
    clock[0] = 18
    assert r.stats()['entries'] == 1
    clock[0] = 20
    with pytest.raises(RuntimeError, match='expired'):
        r.resolve(handle)
    assert r.stats()['entries'] == 0
    assert torch.equal(admitted, torch.ones(64, 2))


@pytest.mark.parametrize('entries,nbytes,error', [(1, 4096, 'entry quota'), (3, 512, 'byte quota')])
def test_quota_failure_does_not_evict_live_memory(entries, nbytes, error):
    r = fresh_registry(entries=entries, nbytes=nbytes)
    old = publish(r, torch.ones(64, 2), op='old')
    with pytest.raises(RuntimeError, match=error):
        publish(r, torch.zeros(64, 2), op='excess')
    assert torch.equal(r.resolve(old).tensor, torch.ones(64, 2))
    assert r.stats()['entries'] == 1


def test_release_replay_preserves_distinct_new_operation():
    r = fresh_registry()
    old = publish(r, torch.ones(64, 2))
    released = r.release(op_id='release-test', handle=old)
    # New logical operations MUST use fresh IDs, as the client already does.
    # Reusing the old publish ID would reproduce the old public descriptor.
    new = publish(r, torch.ones(64, 2), op='next-generation')
    assert old != new
    assert r.release(op_id='release-test', handle=old) == released
    assert r.resolve(new).tensor.shape == (64, 2)


def test_peer_publication_failure_reclaims_local_candidate():
    r = fresh_registry()
    with pytest.raises(RuntimeError, match='publication failed'):
        collective_publish(r, tensor=torch.ones(64, 2), op_id='failure',
            request_digest='sha256:' + 'b' * 64,
            gather=lambda value: [value, dict(success=False, message='peer capacity')])
    assert r.stats()['entries'] == 0


def test_failed_snapshot_write_preserves_previous_durable_file(engine, monkeypatch, tmp_path):
    c = client(engine)
    value = build(c, [3] * 1024)['memory_embeddings']
    saved = c.save_memory(value)
    c.checkpoint_saved({saved['snapshot']})
    path = next((tmp_path / 'remote-snapshots').rglob('*.pt'))
    before = path.read_bytes()

    def broken_save(*args, **kwargs):
        raise OSError('injected disk failure')

    with monkeypatch.context() as patch:
        patch.setattr(torch, 'save', broken_save)
        with pytest.raises(RuntimeError, match='injected disk failure'):
            memory_operation(engine.registry, dict(operation='save', handle=value.descriptor),
                             gather=lambda x: [x])
    assert path.read_bytes() == before
    assert not list(path.parent.glob('*.tmp'))
    restored = c.load_memory(saved)
    assert restored.shape == value.shape
    c.checkpoint_saved([])
    c.release_memory(value)
    c.release_memory(restored)


@pytest.mark.parametrize('partial', [False, True])
def test_stateful_sse_interruption_fails_without_partial_resume(tmp_path, partial):
    from test_stack_streaming import Client, serve, frame
    frames = [frame([42])] if partial else []
    with serve(frames) as (url, requests):
        c = Client(url)
        c.server_memory = True
        c.configure_generation_streaming(artifact_dir=tmp_path)
        with pytest.raises(RuntimeError, match='explicit recovery required'):
            c._post('/generate', dict(rid='fixture', input_ids=[1], sampling_params={'max_new_tokens': 20}))
    assert len(requests) == 1


def test_streaming_uses_authenticated_client_transport(tmp_path, monkeypatch):
    from test_stack_streaming import Client, serve, frame
    with serve([frame([42], finish={'type': 'stop'})]) as (url, requests):
        c = Client(url)
        c.server_memory = True
        c.configure_generation_streaming(artifact_dir=tmp_path)
        opened = []
        original = c._open_request
        def observed(request, *, timeout):
            opened.append(request.full_url)
            return original(request, timeout=timeout)
        monkeypatch.setattr(c, '_open_request', observed)
        result = c._post('/generate', dict(rid='fixture', input_ids=[1], sampling_params={'max_new_tokens': 20}))
        assert result['output_ids'] == [42]
        assert opened == [url + '/generate']
