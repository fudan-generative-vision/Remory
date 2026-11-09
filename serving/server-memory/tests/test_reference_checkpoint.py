import copy
import os
import pickle
from pathlib import Path
import pytest
from test_remory_server_memory import engine, client, build
from shared_runtime.server_memory_checkpoint import save_reference_checkpoint, load_reference_checkpoint, TAG


def test_checkpoint_commit_and_fresh_handles(engine, tmp_path):
    c = client(engine)
    value = build(c, [3] * 300)['memory_embeddings']
    path = tmp_path / 'session/checkpoint.pkl'
    save_reference_checkpoint(path, {'memory':value, 'again':value}, c)
    payload = pickle.loads(path.read_bytes())
    assert TAG in payload['state']['memory'] and path.stat().st_mode & 0o777 == 0o600
    restored = load_reference_checkpoint(path, c)
    assert restored['memory'] is restored['again']
    assert restored['memory'].descriptor['handle_id'] != value.descriptor['handle_id']
    assert restored['memory'].shape == value.shape


def test_failed_atomic_replace_preserves_old_journal_and_roots(engine, tmp_path, monkeypatch):
    c = client(engine);old = build(c, [3] * 300)['memory_embeddings'];path = tmp_path / 'checkpoint.pkl'
    save_reference_checkpoint(path, {'memory':old}, c);original = path.read_bytes()
    snapshots = set(c._checkpoint_snapshots)
    new = build(c, [4] * 300)['memory_embeddings']
    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('injected disk failure')))
        with pytest.raises(OSError, match='injected disk failure'):
            save_reference_checkpoint(path, {'memory':new}, c)
    assert path.read_bytes() == original and c._checkpoint_snapshots == snapshots
    assert old.descriptor['handle_id'] in c.owned_memory()
    assert load_reference_checkpoint(path,c)['memory'].shape == old.shape
    assert not list(path.parent.glob('.remory-checkpoint-*'))


def test_restore_rejects_different_backend_and_public_file(engine, tmp_path):
    c=client(engine);p=tmp_path/'checkpoint.pkl'
    save_reference_checkpoint(p,{'memory':build(c,[3]*100)['memory_embeddings']},c)
    other=client(engine);other.base_url='http://different/backends/worker'
    with pytest.raises(ValueError,match='original pinned backend'):load_reference_checkpoint(p,other)
    assert not other.owned_memory()
    p.chmod(0o644)
    with pytest.raises(ValueError,match='private owned'):load_reference_checkpoint(p,c)


def test_fixed_equivalence_requires_exact_proof_and_identity(monkeypatch, tmp_path):
    from context_compression import h20_equivalence as e
    root=Path(__file__).resolve().parents[1]
    cfg=dict(residual=dict(block_tokens=1024,compression_ratio=16,memory_budget=4096),compressor=dict(target_hidden_size=5120))
    info=dict(remory_artifact_identity=e.REMOTE,specforge_gpu_memory_handle={'hook_abi_digest':e.HOOK})
    monkeypatch.delenv('REMORY_H20_VERIFIED_PAIR',raising=False)
    assert not e.verified_pair(e.LOCAL,info,cfg)
    monkeypatch.setenv('REMORY_H20_VERIFIED_PAIR',e.POLICY)
    proof=root/'docs/online250-tensor-equivalence.json'
    monkeypatch.setenv('REMORY_H20_EQUIVALENCE_PROOF',str(proof))
    assert e.verified_pair(e.LOCAL,info,cfg)
    altered=copy.deepcopy(info);altered['remory_artifact_identity']['contract_sha256']='wrong'
    assert not e.verified_pair(e.LOCAL,altered,cfg)
    bad=tmp_path/'proof.json';bad.write_bytes(proof.read_bytes()+b' ')
    monkeypatch.setenv('REMORY_H20_EQUIVALENCE_PROOF',str(bad))
    assert not e.verified_pair(e.LOCAL,info,cfg)
