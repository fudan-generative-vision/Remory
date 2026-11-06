"""Public weights must load strictly and retain the verified native identity."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import runpy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/recursive_memory/stack/payload/workspace"))
from context_compression import remory_public_release as public
from context_compression import qwen38_residual_serving_contract as contract
from context_compression.qwen38_residual_sglang_hook import _load_summary_residual_compressor


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_public_snapshot_and_derived_contract_verify_actual_bytes(tmp_path, monkeypatch):
    # A small fixture exercises the complete format boundary without rehashing
    # the real 8 GB release on every test run. Production pins remain fixed.
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    public_config = {"fixture": "public configuration"}
    save_file({"weight": torch.arange(6, dtype=torch.float32)}, snapshot / "model.safetensors")
    monkeypatch.setitem(public.RELEASE, "config_sha256", contract.canonical_sha256(public_config))
    monkeypatch.setitem(public.RELEASE, "safetensors_sha256",
                        contract.file_sha256(snapshot / "model.safetensors"))
    manifest = {"schema": "remory.inference.v1", **{
        k: public.RELEASE[k] for k in ("config_sha256", "compressor_sha256", "safetensors_sha256")}}
    write_json(snapshot / "config.json", public_config)
    write_json(snapshot / "manifest.json", manifest)
    assert public.verify_public_snapshot(snapshot) == manifest

    serving = tmp_path / "serving"
    serving.mkdir()
    config = json.loads((ROOT / "handoff/h20/compressor-metadata/config.json").read_text())
    derived = json.loads((ROOT / "handoff/h20/compressor-metadata/manifest.json").read_text())
    derived.update(compressor_file="model.safetensors", public_revision=public.RELEASE["revision"],
                   safetensors_sha256=public.RELEASE["safetensors_sha256"])
    write_json(serving / "config.json", config)
    write_json(serving / "manifest.json", derived)
    (serving / "model.safetensors").symlink_to(snapshot / "model.safetensors")
    assert contract.load_residual_serving_contract(serving, verify_weights=True) == config

    write_json(snapshot / "config.json", {"fixture": "changed"})
    with pytest.raises(ValueError, match="configuration checksum"):
        public.verify_public_snapshot(snapshot)
    write_json(snapshot / "config.json", public_config)
    (snapshot / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="safetensors checksum"):
        public.verify_public_snapshot(snapshot)
    with pytest.raises(ValueError, match="compressor checksum"):
        contract.load_residual_serving_contract(serving, verify_weights=True)


@pytest.mark.parametrize("field,value", [
    ("compressor_sha256", "c2c6efde58350ddb9fb98c7bdaa0e8bb0576be46581751ecd70cc198e67a1764"),
    ("safetensors_sha256", hashlib.sha256(b"another export").hexdigest()),
    ("public_revision", "f24de2ac3996da36678cd5d196a7dc0dbbc3aa34"),
    ("compressor_file", "../model.safetensors"),
])
def test_export_mapping_cannot_relabel_another_release(field, value):
    manifest = dict(compressor_file="model.safetensors", public_revision=public.RELEASE["revision"],
                    compressor_sha256=public.RELEASE["compressor_sha256"],
                    safetensors_sha256=public.RELEASE["safetensors_sha256"])
    public.validate_serving_export(manifest)
    manifest[field] = value
    with pytest.raises(ValueError, match="untrusted public export"):
        public.validate_serving_export(manifest)


def test_relocated_architecture_does_not_need_source_host_path(tmp_path):
    config = {"dflash2_architecture_config": "/missing/source/host/dflash2_config.json"}
    adjacent = tmp_path / "dflash2_config.json"
    shutil.copyfile(public.ARCHITECTURE, adjacent)
    assert public.architecture_path(tmp_path, config) == adjacent
    values = json.loads(adjacent.read_text())
    values["num_hidden_layers"] = 99
    write_json(adjacent, values)
    with pytest.raises(ValueError, match="architecture configuration differs"):
        public.architecture_path(tmp_path, config)


def test_client_upgrade_defaults_do_not_retarget_saved_sessions():
    settings = runpy.run_path(str(ROOT / "tools/client_entry.py"))["client_settings"]
    default = {"endpoint": "http://127.0.0.1:30290", "runtime_file": "/new/runtime.json"}
    assert settings({}, {}, default) == (default["endpoint"], default["runtime_file"])
    assert settings({}, {}, default, resuming=True) == ("", None)
    saved = {"server_memory": {"schema": 1, "endpoint": "https://old.example/backends/a",
                               "runtime_file": "/old/runtime.json"}}
    assert settings(saved, {}, default, resuming=True) == ("https://old.example/backends/a", "/old/runtime.json")
    explicit = {"REMORY_SGLANG_URL": "https://explicit.example", "REMORY_RUNTIME_FILE": "/explicit/runtime.json"}
    assert settings({}, explicit, default) == (explicit["REMORY_SGLANG_URL"], explicit["REMORY_RUNTIME_FILE"])


@pytest.mark.parametrize("filename", ["compressor.pt", "model.safetensors"])
def test_compressor_loader_preserves_weights_buffers_and_strict_keys(tmp_path, monkeypatch, filename):
    class TinyCompressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)
            self.register_buffer("output_embedding_rms", torch.tensor(0.25))
            self.rotary_emb = torch.nn.Identity()

    calls = []

    def build(config, *, device, checkpoint):
        calls.append((device, checkpoint))
        with torch.device(device):
            return TinyCompressor()

    monkeypatch.setattr(contract, "build_summary_residual_compressor", build)
    state = {"proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
             "output_embedding_rms": torch.tensor(0.25)}
    path = tmp_path / filename
    save = save_file if filename.endswith(".safetensors") else torch.save
    save(state, path)
    metadata = SimpleNamespace(raw_config={}, checkpoint=tmp_path, state_path=path)
    model = _load_summary_residual_compressor(metadata, "cpu", torch.float32)
    assert calls == [("meta", tmp_path)]
    assert not model.training and not model.proj.weight.requires_grad
    assert set(model.state_dict()) == set(state)
    for key, value in model.state_dict().items():
        assert not value.is_meta and torch.equal(value, state[key])
    del state["output_embedding_rms"]
    save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_summary_residual_compressor(metadata, "cpu", torch.float32)
