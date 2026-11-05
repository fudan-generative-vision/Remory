from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from remory.codex.legacy_bridge import LegacyBridge
from remory.pyramid import plan_pyramid
from remory.server import create_app
from remory.store import MemoryStore

CONFIG = {'schema': 'legacy-test', 'target_model': '/official/local-model',
    'compressor': {'target_hidden_size': 3, 'max_depth': 8},
    'residual': {'block_tokens': 8, 'compression_ratio': 2, 'memory_budget': 8},
    'summary_contract': {'end_token_id': 99, 'student_summary_before_ids': [20],
        'student_summary_after_ids': [21], 'student_memory_before_ids': [22],
        'student_memory_after_ids': [23]}}


class LegacyClient:
    base_url = 'http://legacy-test'

    def __init__(self):
        self._residual_config = deepcopy(CONFIG)
        self.calls = []
        self.fail = False
        self.changed = False

    def attest_residual_server(self, checkpoint):
        return {'context_limit': 1000, 'checkpoint': checkpoint, 'encoder': 'new' if self.changed else '14925'}

    def build_residual_memory(self, prefix, history, summary, **kwargs):
        self.calls.append((prefix, history, summary, kwargs))
        if self.fail:
            raise RuntimeError('old encoder failed')
        depths = [0] * ((len(history) + 7) // 8)
        for position, depth in zip(kwargs.get('source_memory_positions', []),
                                   kwargs.get('source_memory_depths', [])):
            index = (position - len(prefix)) // 8
            depths[index] = max(depths[index], depth)
        plan = plan_pyramid(len(history), budget=kwargs['memory_budget'], block=8, ratio=2,
                            max_depth=None, leaf_input_depths=depths)
        return {'memory_embeddings': np.ones((plan.soft_tokens, 3), dtype=np.float32),
                'receipt': {'slot_depths': [1] * plan.soft_tokens, 'old_encoder_receipt': True}}

    def generate_with_memory(self, ids, **kwargs):
        self.calls.append(('memory', ids, kwargs))
        return SimpleNamespace(token_ids=(42,), meta_info={'completion_tokens': 1}, finish_reason='stop')

    def generate_discrete(self, ids, **kwargs):
        self.calls.append(('discrete', ids, kwargs))
        return SimpleNamespace(token_ids=(42,), meta_info={}, finish_reason='stop')


@pytest.fixture
def bridge(tmp_path):
    client = LegacyClient()
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: 'decoded')
    return LegacyBridge(client, tokenizer, tmp_path / 'checkpoint',
                        MemoryStore(tmp_path / 'memories.sqlite'), tensor_factory=np.asarray)


def compact(bridge, **kwargs):
    args = {'owner': 'session', 'prefix_ids': [1, 2], 'history_ids': [3, 4, 5], 'summary_ids': [6]}
    args.update(kwargs)
    return bridge.compact(**args)


def test_previous_uses_exact_envelope_and_sparse_memory(bridge):
    first = compact(bridge)
    next_memory = compact(bridge, previous=first.handle, history_ids=[7, 8])
    prefix, history, summary, kwargs = bridge.backend.client.calls[-1]
    assert prefix == [1, 2]
    assert history == [20, 6, 21, 22] + [99] * 4 + [23, 7, 8]
    assert kwargs['source_memory_positions'] == [6, 7, 8, 9]
    assert kwargs['source_memory_depths'] == [1] * 4
    assert kwargs['contract'] == CONFIG['summary_contract']
    assert next_memory.receipt['previous'] == first.handle
    assert bridge.store.get('session', first.handle).summary_ids == (6,)
    result = bridge.generate(owner='session', handle=next_memory.handle,
                             continuation_ids=[88], max_new_tokens=2)
    assert result.text == 'decoded'
    assert bridge.backend.client.calls[-1][0] == 'memory'
    assert bridge.backend.client.calls[-1][1][-1] == 88


def test_branch_reencodes_without_deleting_or_mixing_owners(tmp_path):
    client = LegacyClient()
    client._residual_config['residual']['memory_budget'] = 16
    bridge = LegacyBridge(client, None, tmp_path / 'checkpoint',
                          MemoryStore(tmp_path / 'store.sqlite'), tensor_factory=np.asarray)
    parent = compact(bridge)
    child = compact(bridge, prefix_ids=[100], summary_ids=[55])
    merged = compact(bridge, previous=parent.handle, branches=[child.handle])
    _, history, _, kwargs = bridge.backend.client.calls[-1]
    assert 55 in history and len(kwargs['source_memory_positions']) == 8
    assert merged.receipt['branches'] == [child.handle]
    assert bridge.store.get('session', child.handle).prefix_ids == (100,)
    with pytest.raises(KeyError):
        compact(bridge, owner='foreign', branches=[child.handle])
    with pytest.raises(ValueError, match='twice'):
        compact(bridge, branches=[child.handle, child.handle])


def test_bad_prefix_and_identity_fail_before_encoding(bridge):
    first = compact(bridge)
    count = len(bridge.backend.client.calls)
    with pytest.raises(ValueError, match='prefix'):
        compact(bridge, previous=first.handle, prefix_ids=[9])
    bridge.backend.client.changed = True
    with pytest.raises(RuntimeError, match='identity'):
        compact(bridge, previous=first.handle)
    with pytest.raises(RuntimeError, match='identity'):
        bridge.generate(owner='session', handle=first.handle, max_new_tokens=1)
    assert len(bridge.backend.client.calls) == count


def test_failed_build_and_failed_store_keep_old_checkpoint(bridge, monkeypatch):
    first = compact(bridge)
    bridge.backend.client.fail = True
    with pytest.raises(RuntimeError, match='encoder'):
        compact(bridge, previous=first.handle)
    bridge.backend.client.fail = False
    def fail_store(**kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(bridge.store, 'put', fail_store)
    with pytest.raises(OSError, match='disk full'):
        compact(bridge, previous=first.handle)
    assert bridge.store.get('session', first.handle).summary_ids == (6,)
    with bridge.store._connect() as db:
        assert db.execute('SELECT count(*) FROM memories').fetchone()[0] == 1


def test_public_api_restart_delete_and_no_online_claim(bridge):
    http = TestClient(create_app(bridge))
    info = http.get('/v1/remory/info').json()
    assert info['config'] == CONFIG and 'online_training' not in info
    headers = {'X-Remory-Session': 'session'}
    body = {'prefix_ids': [1], 'history_ids': [2, 3], 'summary_ids': [4]}
    response = http.post('/v1/compact', headers=headers, json=body)
    assert response.status_code == 200
    handle = response.json()['handle']
    restored = LegacyBridge(bridge.backend.client, bridge.backend.tokenizer,
        bridge.backend.checkpoint, MemoryStore(bridge.store.path), tensor_factory=np.asarray)
    assert restored.store.get('session', handle).identity == restored.contract.identity
    assert http.post('/generate', headers=headers, json={'input_ids': [9],
        'remory': {'handle': handle}, 'sampling_params': {'max_new_tokens': 1}}).status_code == 200
    assert http.post('/v1/compact', headers=headers, json={**body, 'encoder_version': 'new'}).status_code == 400
    assert http.delete('/v1/memories/' + handle, headers=headers).status_code == 200
    with pytest.raises(KeyError):
        restored.store.get('session', handle)


def test_overflow_and_bad_geometry_do_not_commit(bridge):
    with pytest.raises(ValueError, match='limit'):
        compact(bridge, summary_ids=[7] * 1000)
    assert not bridge.backend.client.calls
    original = bridge.backend.client.build_residual_memory
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result['memory_embeddings'][0, 0] = np.nan
        return result
    bridge.backend.client.build_residual_memory = corrupt
    with pytest.raises(ValueError, match='nonfinite'):
        compact(bridge)
    with bridge.store._connect() as db:
        assert db.execute('SELECT count(*) FROM memories').fetchone()[0] == 0


def test_real_legacy_builder_with_fake_encode_only(tmp_path):
    """Opt-in integration of the existing legacy algorithm, never HTTP/model calls."""
    import os
    import sys
    workspace = os.environ.get('REMORY_TEST_LEGACY_WORKSPACE')
    if not workspace:
        pytest.skip('Set REMORY_TEST_LEGACY_WORKSPACE for offline legacy algorithm integration')
    sys.path.insert(0, workspace)
    import torch
    from context_compression.qwen38_residual_sglang_client import SummaryResidualSGLangClient
    from context_compression.qwen38_residual_sglang_hook import (
        RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY, RESIDUAL_RECURSIVE_CAPABILITY)
    client = SummaryResidualSGLangClient.__new__(SummaryResidualSGLangClient)
    client.base_url = 'http://no-network'
    client.eos_token_id = 99
    client._residual_config = deepcopy(CONFIG)
    client._residual_context_limit = 1000
    client._residual_attestation = {'context_limit': 1000,
        'dynamic_summary_capability': RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY,
        'recursive_compact_capability': RESIDUAL_RECURSIVE_CAPABILITY}
    client.attest_residual_server = lambda checkpoint: deepcopy(client._residual_attestation)
    client._validate_generate_lifecycle_locked = lambda: None
    calls = []
    def fake_encode(ids, **kwargs):
        calls.append((ids, kwargs))
        return torch.full((kwargs['expected_rows'], 3), float(len(calls)))
    client._encode_residual = fake_encode
    bridge = LegacyBridge(client, None, tmp_path / 'checkpoint', MemoryStore(tmp_path / 'store.sqlite'))
    first = compact(bridge, history_ids=list(range(20)))
    assert any(c[1]['operation'] == 'parent' for c in calls)
    calls.clear()
    second = compact(bridge, previous=first.handle, history_ids=[8])
    recursive = next(c for c in calls if c[1]['operation'] == 'recursive_leaves')
    assert torch.equal(recursive[1]['child_memory'], torch.tensor(first.embeddings))
    assert second.receipt['cross_summary_reconditioning']
    assert second.receipt['source_memory_slots'] == len(first.embeddings)
    assert bridge.store.get('session', first.handle).handle == first.handle
