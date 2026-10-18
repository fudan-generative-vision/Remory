import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from remory.online.collect import Pool, valid_sample
from remory.online.common import ENCODER_VERSION, checkpoint, checksum, digest, read, resolve_version, write_json
from remory.online.control import Control
from remory.online.devices import resolve, serving_devices, validate_devices
from remory.server import create_app


def artifact(path, config, data=b'weights'):
    path.mkdir(parents=True)
    (path / 'model.safetensors').write_bytes(data)
    write_json(path / 'config.json', config)
    write_json(path / 'manifest.json', dict(config_sha256=digest(config),
        safetensors_sha256=checksum(path / 'model.safetensors')))
    return checkpoint(path)


@pytest.fixture
def pool(runtime, tmp_path):
    config = {'target_model': 'fixture', 'target_revision': 'pinned'}
    base = tmp_path / 'base'
    artifact(base, config)
    runtime.model_config = config
    runtime.online = Pool(tmp_path / 'pool', config=config, base=base, actor=tmp_path,
        identity=runtime.contract.identity, serving_gpus=['GPU-serving'])
    return runtime.online


def test_capture_requires_durable_client_acceptance_and_preserves_old_memory(runtime, pool):
    app = TestClient(create_app(runtime))
    headers = {'X-Remory-Session': 'session-a'}
    body = dict(prefix_ids=[1, 3], history_ids=list(range(20)), summary_ids=[50, 51],
        capture_opening_ids=[98, 99], encoder_version=pool.meta['base_encoder'])
    response = app.post('/v1/compact', json=body, headers=headers)
    assert response.status_code == 200, response.text
    result = response.json()
    folder = pool.root / 'samples' / result['sample_id']
    with pytest.raises(ValueError, match='installed summary'):
        valid_sample(folder)
    route = '/v1/remory/samples/' + result['sample_id'] + '/accept'
    assert app.post(route, headers={'X-Remory-Session': 'session-b'}).status_code == 404
    assert app.post(route, headers=headers).status_code == 200
    source, summary = valid_sample(folder)
    assert source['source_ids'] == list(range(20))
    assert source['opening_ids'] == [98, 99]
    assert 'target_ids' not in source and 'summary_ids' not in source
    assert summary['ids'] == [50, 51]
    second = app.post('/v1/compact', json={**body, 'previous': result['handle'],
        'history_ids': [71, 72]}, headers=headers).json()
    app.post('/v1/remory/samples/' + second['sample_id'] + '/accept', headers=headers)
    next_folder = pool.root / 'samples' / second['sample_id']
    recursive, _ = valid_sample(next_folder)
    assert recursive['memory_positions'] and recursive['source_ids'][-2:] == [71, 72]
    assert np.array_equal(np.load(next_folder / 'memory.npy'),
        runtime.store.get('session-a', result['handle']).embeddings)
    (next_folder / 'memory.npy').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='tensor changed'):
        valid_sample(next_folder)


def test_failed_encode_never_produces_training_sample(runtime, pool):
    runtime.backend.fail = 'leaves'
    app = TestClient(create_app(runtime))
    r = app.post('/v1/compact', headers={'X-Remory-Session': 'session-a'},
        json=dict(prefix_ids=[1], history_ids=[2, 3], summary_ids=[4], capture_opening_ids=[5]))
    assert r.status_code == 502
    assert not list((pool.root / 'samples').glob('*/frontier.json'))
    assert ENCODER_VERSION.get() is None


def test_default_server_does_not_accept_online_fields(runtime):
    app = TestClient(create_app(runtime))
    r = app.post('/v1/compact', headers={'X-Remory-Session': 'session-a'},
        json=dict(prefix_ids=[1], history_ids=[2], summary_ids=[4], capture_opening_ids=[5]))
    assert r.status_code == 400


def test_checkpoint_selection_is_per_session_and_applied_only_after_commit(runtime, pool, tmp_path):
    info = dict(online_training=pool.meta)
    one = Control(tmp_path / 'one', info, 'http://local')
    two = Control(tmp_path / 'two', info, 'http://local')
    trained = artifact(pool.root / 'checkpoints/step-1/serving', pool.config, b'trained')
    artifact(pool.root / 'checkpoints/incompatible/serving', {'target_model': 'other'}, b'bad')
    artifact(pool.root / 'checkpoints/.saving/serving', pool.config, b'not-published')
    assert len(one.candidates()) == 2
    one.select(trained['id'])
    assert one.selection()['id'] == trained['id']
    assert two.selection()['id'] == pool.meta['base_encoder']
    assert not (one.root / 'active.json').exists()
    one.applied(trained['id'])
    assert not (one.root / 'pending-load.json').exists()
    assert read(one.root / 'active.json')['id'] == trained['id']
    one.select(pool.meta['base_encoder'])
    one.applied(pool.meta['base_encoder'])
    assert one.selection()['id'] == pool.meta['base_encoder']
    # A registry entry cannot redirect the serving worker to arbitrary files.
    outside = artifact(tmp_path / 'outside', pool.config, b'outside')
    write_json(pool.root / 'versions' / (outside['id'] + '.json'), outside)
    with pytest.raises(ValueError, match='outside'):
        resolve_version(pool.root, pool.base, outside['id'])
    (Path(trained['path']) / 'model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        one.select(trained['id'])


def test_gpu_ids_follow_visible_mapping_and_reject_serving_overlap(monkeypatch):
    available = {'0': {'uuid': 'GPU-a'}, '1': {'uuid': 'GPU-b'}, '2': {'uuid': 'GPU-c'}}
    monkeypatch.setattr('remory.online.devices.inventory', lambda: available)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,0')
    assert serving_devices(0, 2) == ['GPU-c', 'GPU-a']
    assert resolve(['1'], available) == ['GPU-b']
    with pytest.raises(ValueError, match='exclude inference'):
        validate_devices(['GPU-a'], ['GPU-a'])
    with pytest.raises(ValueError, match='distinct'):
        resolve(['0', 'GPU-a'], available)


def test_configure_updates_stale_server_address_only_when_worker_is_stopped(pool, tmp_path, monkeypatch):
    import remory.online.control as module
    monkeypatch.setattr(module, 'inventory', lambda: {'2': {'uuid': 'GPU-training'}})
    monkeypatch.setattr(module, 'validate_devices', lambda *_: None)
    one = Control(tmp_path / 'session', {'online_training': pool.meta}, 'http://first')
    one.configure(gpus='2')
    monkeypatch.setattr(module, 'alive', lambda _: True)
    one.configure()  # An unchanged pool can be joined while training.
    one.url = 'http://second'
    with pytest.raises(ValueError, match='Stop the shared trainer'):
        one.configure()
    assert read(pool.root / 'config.json')['url'] == 'http://first'
    monkeypatch.setattr(module, 'alive', lambda _: False)
    one.configure()
    assert read(pool.root / 'config.json')['url'] == 'http://second'


def test_provider_accepts_samples_only_after_persisting_the_memory(tmp_path):
    from test_codex import Tokenizer
    from remory.codex.provider import Provider
    calls, applied = [], []
    info = dict(identity='fixture', config={'summary_contract': {'generation_suffix_ids': [999]}})
    selected = 'a' * 64
    control = SimpleNamespace(selection=lambda: {'id': selected}, applied=applied.append)
    def backend(request):
        body = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path == '/v1/compact':
            assert body['encoder_version'] == selected
            assert body['capture_opening_ids'] == list(b'[assistant]<think>')
            return httpx.Response(200, json={'handle': 'rm_saved', 'receipt': {}, 'sample_id': 'b' * 32})
        if request.url.path.endswith('/accept'):
            states = [read(p) for p in tmp_path.glob('thread-*.json')]
            assert any('remory.v1.rm_saved' in s['checkpoints'] for s in states)
            return httpx.Response(200, json={'accepted': True})
        return httpx.Response(200, json={'text': 'Keep the task.',
            'meta_info': {'finish_reason': {'type': 'stop'}}})
    provider = Provider('http://server', info, Tokenizer(), tmp_path, session_id='session-a',
        online_control=control, transport=httpx.MockTransport(backend))
    req = {'input': [{'role': 'user', 'content': 'Continue this task.'}, {'type': 'compaction_trigger'}]}
    response = provider.handle(req)
    assert len(calls) == 3 and applied == [selected]
    assert provider.handle(req) == response and len(calls) == 3


def test_hook_loads_only_selected_weights_and_restores_base(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    save_file = pytest.importorskip('safetensors.torch').save_file
    from remory.backends.sglang_hook import ResidualHook
    config = {'target_model': 'fixture'}
    pool, base = tmp_path / 'pool', tmp_path / 'base'
    model = torch.nn.Linear(2, 2, bias=False)
    def weights(path, value):
        path.mkdir(parents=True)
        save_file({'weight': torch.full((2, 2), value)}, str(path / 'model.safetensors'))
        write_json(path / 'config.json', config)
        write_json(path / 'manifest.json', dict(config_sha256=digest(config),
            safetensors_sha256=checksum(path / 'model.safetensors')))
        return checkpoint(path)
    initial = weights(base, 1.)
    trained = weights(pool / 'checkpoints/step-1/serving', 2.)
    write_json(pool / 'versions' / (trained['id'] + '.json'), trained)
    monkeypatch.setenv('REMORY_ENCODER_DIRECTORY', str(pool))
    hook = ResidualHook({'checkpoint': str(base)})
    hook.compressor, hook.base_version = model, initial['id']
    hook.select_encoder(trained['id'])
    assert torch.equal(model.weight, torch.full((2, 2), 2.))
    hook.select_encoder(None)
    assert torch.equal(model.weight, torch.full((2, 2), 1.))
    with pytest.raises(ValueError, match='Unknown'):
        hook.select_encoder('f' * 64)
    assert torch.equal(model.weight, torch.full((2, 2), 1.))
