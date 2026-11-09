"""CPU protocol oracles: real hook/registry/client, deterministic tiny model.

These tests never launch a model server or allocate a GPU. Actual CUDA/TP
deployment still requires the separate idle-server smoke test.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

PAYLOAD = Path(__file__).resolve().parents[1] / "src/recursive_memory/stack/payload"
sys.path[:0] = [str(PAYLOAD / "workspace"), str(PAYLOAD / "sglang_overlay/python")]
from context_compression.remory_memory import RemoteMemory, CAPABILITY
from context_compression.remory_sglang_client import ServerResidualSGLangClient
from context_compression.qwen38_residual_sglang_client import SummaryResidualSGLangClient
from context_compression.qwen38_residual_sglang_hook import (
    SummaryResidualSGLangHook, RESIDUAL_RECURSIVE_CAPABILITY, RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY,
)
from context_compression.sglang_compressor_hook import CUSTOM_PARAM_KEY, GPU_MEMORY_HANDLE_META_KEY
from sglang.srt.memory_handle_registry import TPMemoryHandleRegistry
from sglang.srt.remory_memory import collective_publish, memory_operation


def registry(epoch="epoch1", **kwargs):
    return TPMemoryHandleRegistry(server_epoch=epoch, tp_rank=0, ttl_seconds=3600,
        max_entries=256, max_bytes=16 << 20, hook_abi_digest="sha256:" + "a" * 64,
        require_cuda=False, **kwargs)


def publish(r, tensor, op="first"):
    return collective_publish(r, tensor=tensor, op_id=op, request_digest="sha256:" + "b" * 64,
                              gather=lambda x: [x])


class TinyCompressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, features, mask, *, summary_features, input_depth, **kwargs):
        mean = (features * mask[..., None]).sum(1, keepdim=True) / mask.sum(1)[:, None, None]
        value = mean + summary_features.mean(1, keepdim=True) + input_depth
        return SimpleNamespace(embeddings=value.expand(features.shape[0], 64, 2),
                               attention_mask=torch.ones(features.shape[0], 64, dtype=torch.bool))


class Engine:
    def __init__(self, root, epoch="epoch1"):
        self.registry = registry(epoch)
        self.hook = object.__new__(SummaryResidualSGLangHook)
        self.hook.metadata = SimpleNamespace(target_hidden_size=2, compression_ratio=16)
        self.hook.residual_contract = SimpleNamespace(max_depth=8, block_tokens=1024)
        self.hook.hook_abi_digest = self.registry.hook_abi_digest
        self.hook._memory_handle_registry = self.registry
        self.hook.compressor = TinyCompressor()
        self.hook._validate_encode_forward = lambda inputs, batch, **kwargs: (None, [inputs[1]])
        self.requests = []
        self.fail = None

    def generate(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        payload = request["sampling_params"]["custom_params"][CUSTOM_PARAM_KEY]
        if self.fail == payload["residual"]["operation"]:
            raise RuntimeError("injected encoder failure")
        ids = torch.tensor(request["input_ids"])
        hidden = torch.stack((ids.float(), ids.float() / 2), dim=1)
        req = SimpleNamespace()
        if "memory_handle" in payload:
            req.memory_handle_tensor_ref = self.registry.resolve(payload["memory_handle"]).tensor
            hidden[payload["memory_positions"]] = req.memory_handle_tensor_ref
        elif "memory_embeddings" in payload:
            hidden[payload["memory_positions"]] = torch.tensor(payload["memory_embeddings"])
        if "summary_handle" in payload["residual"]:
            descriptor = payload["residual"]["summary_handle"]
            req.remory_summary_descriptor = descriptor
            req.remory_summary_tensor_ref = self.registry.resolve(descriptor).tensor
        batch = SimpleNamespace(sampling_info=SimpleNamespace(custom_params=[{CUSTOM_PARAM_KEY: payload, "__req__": req}]))
        output = self.hook(None, (ids, hidden, None, batch), SimpleNamespace(hidden_states=None, customized_info=None))
        meta = {}
        if output.hidden_states is not None:
            meta["hidden_states"] = [output.hidden_states.tolist()]
        if output.customized_info and GPU_MEMORY_HANDLE_META_KEY in output.customized_info:
            meta[GPU_MEMORY_HANDLE_META_KEY] = output.customized_info[GPU_MEMORY_HANDLE_META_KEY][0]
        return {"meta_info": meta}


def client(engine, *, remote=True):
    cls = ServerResidualSGLangClient if remote else SummaryResidualSGLangClient
    c = cls("http://engine/backends/gpu0")
    c.eos_token_id = 1
    c._memory_epoch = engine.registry.server_epoch
    c._memory_hook_digest = engine.registry.hook_abi_digest
    c._residual_context_limit = 262143
    c._residual_config = dict(summary_contract=dict(student_summary_before_ids=[20], student_summary_after_ids=[21]),
        residual=dict(memory_budget=4096, block_tokens=1024, compression_ratio=16),
        compressor=dict(target_hidden_size=2, max_depth=8))
    c._residual_attestation = dict(dynamic_summary_capability=RESIDUAL_DYNAMIC_SUMMARY_CAPABILITY,
        recursive_compact_capability=RESIDUAL_RECURSIVE_CAPABILITY)
    c._isolated_cache_policy_active = lambda: True
    c._processor_string = lambda: "oracle"
    c._require_cache_bypass_receipt = lambda x: None
    c._post_generate = engine.generate

    def post(route, value, **kwargs):
        if route == "/release_memory_handle":
            result = engine.registry.release(op_id=value["op_id"], handle=value["handle"])
            return dict(success=True, message="", **result)
        return dict(success=True, result=memory_operation(engine.registry, value["payload"], gather=lambda x: [x]))
    c._post = post
    return c


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("REMORY_MEMORY_DIR", str(tmp_path / "remote-snapshots"))
    return Engine(tmp_path)


def build(c, history, **kwargs):
    return c.build_residual_memory([7, 8], history, [11, 12], contract=c._residual_config["summary_contract"], **kwargs)


def test_unbounded_frontier_exceeds_old_4096_cap_without_tensor_wire(engine):
    c = client(engine)
    result = build(c, [3] * (65 * 1024), memory_budget=None)
    memory = result["memory_embeddings"]
    assert isinstance(memory, RemoteMemory) and memory.shape == (4160, 2)
    assert result["receipt"]["memory_budget"] is None
    assert result["receipt"]["parent_count"] == 0
    assert engine.registry.stats()["entries"] == 1  # summary was reclaimed
    for request in engine.requests:
        assert request["return_hidden_states"] is False
        wire = json.dumps(request)
        assert "memory_embeddings" not in wire and "summary_features" not in wire
    assert len(json.dumps(memory.to_json())) < 2000


def test_recursive_handle_path_matches_tensor_reference(engine):
    remote, legacy = client(engine), client(engine, remote=False)
    first = build(remote, [3] * 1025)
    reference = build(legacy, [3] * 1025)
    handle = first["memory_embeddings"]
    torch.testing.assert_close(engine.registry.resolve(handle.descriptor).tensor, reference["memory_embeddings"])
    positions = list(range(2, 130))
    source = [1] * 128 + [9] * 1500
    second = build(remote, source, source_memory=handle, source_memory_positions=positions, source_memory_depths=[1] * 128)
    expected = build(legacy, source, source_memory=reference["memory_embeddings"],
                     source_memory_positions=positions, source_memory_depths=[1] * 128)
    torch.testing.assert_close(engine.registry.resolve(second["memory_embeddings"].descriptor).tensor,
                               expected["memory_embeddings"])
    assert engine.registry.stats()["entries"] == 2  # old survives until install
    remote.release_memory(handle)
    assert engine.registry.stats()["entries"] == 1


def test_failed_build_reclaims_intermediate_but_keeps_old_memory(engine):
    c = client(engine)
    old = build(c, [3] * 1024)["memory_embeddings"]
    engine.fail = "recursive_leaves"
    with pytest.raises(RuntimeError, match="injected"):
        build(c, [1] * 64 + [4] * 1024, source_memory=old,
              source_memory_positions=list(range(2, 66)), source_memory_depths=[1] * 64)
    assert engine.registry.stats()["entries"] == 1
    assert engine.registry.resolve(old.descriptor).tensor.shape == (64, 2)


def test_released_entry_does_not_invalidate_admitted_request(engine):
    c = client(engine)
    old = build(c, [3] * 1024)["memory_embeddings"]
    request_ref = engine.registry.resolve(old.descriptor).tensor
    c.release_memory(old)
    assert engine.registry.stats()["entries"] == 0
    assert torch.isfinite(request_ref).all() and request_ref.shape == (64, 2)
    with pytest.raises(Exception, match="missing|expired"):
        engine.registry.resolve(old.descriptor)


def test_remote_snapshot_survives_engine_epoch_change(engine, tmp_path):
    c = client(engine)
    memory = build(c, [3] * 1024)["memory_embeddings"]
    expected = engine.registry.resolve(memory.descriptor).tensor.clone()
    snapshot = c.save_memory(memory, preserve=True)
    c.close_memory()
    restarted = Engine(tmp_path, epoch="epoch2")
    restored = client(restarted).load_memory(snapshot)
    assert restored.descriptor["server_epoch"] == "epoch2"
    torch.testing.assert_close(restarted.registry.resolve(restored.descriptor).tensor, expected)


def test_compose_is_server_local_and_owns_its_storage(engine):
    c = client(engine)
    a = build(c, [3] * 1024)["memory_embeddings"]
    b = build(c, [4] * 1024)["memory_embeddings"]
    combined = c.compose_memory([a, b])
    expected = torch.cat([engine.registry.resolve(v.descriptor).tensor for v in (a, b)])
    c.release_memory(a)
    c.release_memory(b)
    torch.testing.assert_close(engine.registry.resolve(combined.descriptor).tensor, expected)
    assert engine.registry.stats()["entries"] == 1


def test_tp_publication_rolls_back_if_another_rank_fails():
    r = registry()
    with pytest.raises(RuntimeError, match="publication failed"):
        collective_publish(r, tensor=torch.ones(64, 2), op_id="failed", request_digest="sha256:" + "b" * 64,
            gather=lambda value: [value, dict(success=False, message="quota exceeded")])
    assert r.stats()["entries"] == 0


def test_gateway_requires_affinity_for_nested_handles():
    from recursive_memory.stack.gateway import needs_pin
    assert needs_pin("/generate", {"sampling_params": {"custom_params": {
        CUSTOM_PARAM_KEY: {"memory_handle_publish": {"op_id": "test"}}}}})
    assert needs_pin("/remory/memory", {"payload": {"operation": "compose"}})
    assert not needs_pin("/generate", {"input_ids": [1, 2]})


def make_generator(tmp_path, engine, name):
    import runpy
    helpers = runpy.run_path(str(Path(__file__).with_name("test_subagent_folding.py")))
    generator, text_client = helpers["make"](tmp_path, name)
    c = client(engine)
    c._residual_config["summary_contract"] = generator.summary_contract
    c.generate_discrete = text_client.generate_discrete
    c.generate_with_memory = text_client.generate_with_memory
    generator.client = c
    generator.server_memory = True
    generator.residual["memory_budget"] = None
    generator._test_checkpoint = tmp_path / name / "checkpoint.pkl"
    return generator, helpers


def compact(generator, messages):
    summary = generator.summarize(messages=messages, episode_id=generator.episode_id,
                                   round_index=len(messages)//2)
    generator.install(episode_id=generator.episode_id, target_completed_rounds=len(messages)//2-1,
                      summary=summary)
    from shared_runtime.server_memory_checkpoint import save_reference_checkpoint
    save_reference_checkpoint(generator._test_checkpoint, {"memory": generator.memory}, generator.client)
    generator.commit_remote_retirements()
    return summary


def test_main_replaces_old_generation_and_disk_snapshot(engine, tmp_path):
    g, helpers = make_generator(tmp_path, engine, "main")
    messages = helpers["history"]()
    compact(g, messages)
    first = g.memory
    for _ in range(4):
        messages += [{"role": "assistant", "content": "NEXT"}, {"role": "tool", "content": "RESULT"}]
        compact(g, messages)
        assert engine.registry.stats()["entries"] == 1
        assert len(list((tmp_path / "remote-snapshots").rglob("*.pt"))) == 1
        assert len(g.client._owned_memory) == 1
    with pytest.raises(Exception, match="missing|expired"):
        engine.registry.resolve(first.descriptor)
    assert g._context_memory is None


def test_main_failed_replacement_retains_active_generation(engine, tmp_path):
    g, helpers = make_generator(tmp_path, engine, "main")
    messages = helpers["history"]()
    compact(g, messages)
    old = g.memory
    engine.fail = "recursive_leaves"
    messages += [{"role": "assistant", "content": "NEXT"}, {"role": "tool", "content": "RESULT"}]
    with pytest.raises(RuntimeError, match="injected"):
        compact(g, messages)
    assert g.memory is old and g._prepared is None
    assert engine.registry.stats()["entries"] == 1


def test_child_snapshot_contains_only_reference_and_is_consumed(engine, tmp_path):
    from shared_runtime.subagent_memory import export_checkpoint, read_checkpoint
    child, helpers = make_generator(tmp_path, engine, "child")
    session = SimpleNamespace(messages=helpers["history"](), episode_id="child", raw_suffix_images=[])
    path = export_checkpoint(SimpleNamespace(_generator=child), session, tmp_path / "child-fold",
                             child_id="child", parent_id="main")
    assert not Path(path).with_name("memory.pt").exists()
    assert "remote_memory" in json.loads(Path(path).read_text())
    child.client.close_memory()
    assert engine.registry.stats()["entries"] == 0
    main, helpers = make_generator(tmp_path, engine, "main")
    restored = read_checkpoint(path, main, parent_id="main", child_id="child")
    assert isinstance(restored["memory"], RemoteMemory)
    assert main.add_subagent_checkpoint(restored, messages=helpers["history"](), episode_id="main")
    compact(main, helpers["history"]())
    assert main._folded_branches == []
    assert engine.registry.stats()["entries"] == 1


def test_generate_request_uses_handle_and_bypass(engine):
    c = client(engine)
    value = build(c, [3] * 1024)["memory_embeddings"]
    captured = {}
    c._generate = lambda ids, **kwargs: captured.update(kwargs)
    c.generate_with_memory([1] * 66, memory_positions=list(range(2, 66)),
                           memory_embeddings=value, expected_hidden_size=2, max_new_tokens=10)
    params = captured["custom_params"][CUSTOM_PARAM_KEY]
    assert params["memory_handle"] == value.descriptor
    assert "memory_embeddings" not in params
    assert captured["memory_transport_cache_policy"] is None  # inherited isolated bypass


def test_journal_retains_old_snapshot_until_new_commit(engine, tmp_path):
    from shared_runtime.server_memory_checkpoint import pack_memory_state, restore_memory_state
    c = client(engine)
    first = build(c, [3] * 1024)["memory_embeddings"]
    state, snapshots = pack_memory_state({"memory": first, "same": [first]}, c)
    c.checkpoint_saved(snapshots)
    second = build(c, [4] * 2048)["memory_embeddings"]
    c.release_memory(first)
    # A crash here must still be able to load the previously committed journal.
    recovered = restore_memory_state(state, c)
    assert recovered["memory"] is recovered["same"][0]
    c.release_memory(recovered["memory"])
    assert len(list((tmp_path / "remote-snapshots").rglob("*.pt"))) == 2
    new_state, new_snapshots = pack_memory_state({"memory": second}, c)
    c.checkpoint_saved(new_snapshots)
    assert len(list((tmp_path / "remote-snapshots").rglob("*.pt"))) == 1
    assert len(json.dumps(new_state)) < 1000
    c.close_memory()
    restarted = Engine(tmp_path, epoch="epoch2")
    restored = restore_memory_state(new_state, client(restarted))
    assert restored["memory"].descriptor["server_epoch"] == "epoch2"
    assert restored["memory"].shape == second.shape


def test_legacy_checkpoint_is_imported_once_and_rewritten_as_references(engine):
    from shared_runtime.server_memory_checkpoint import pack_memory_state, restore_memory_state, TAG
    c = client(engine)
    tensor = torch.tensor([[1., 2.], [3., 4.]])
    state = restore_memory_state({"memory": tensor, "same": (tensor,)}, c)
    assert state["memory"] is state["same"][0]
    assert engine.registry.stats()["entries"] == 1
    actual = engine.registry.resolve(state["memory"].descriptor).tensor
    torch.testing.assert_close(actual, tensor)
    packed, _ = pack_memory_state(state, c)
    assert TAG in packed["memory"]
    assert "values" not in json.dumps(packed)


def test_cleanup_timeout_keeps_retry_ownership_without_rollback(engine):
    c = client(engine)
    value = build(c, [3] * 1024)["memory_embeddings"]
    post = c._post
    def fail_release(route, payload, **kwargs):
        if route == "/release_memory_handle":
            assert kwargs["request_timeout"] == 10
            raise TimeoutError("lost reply")
        return post(route, payload, **kwargs)
    c._post = fail_release
    with pytest.raises(TimeoutError):
        c.release_memory(value)
    assert value.descriptor["handle_id"] in c._retired_memory
    c._post = post
    c.release_memory(value)
    assert not c._owned_memory and not c._retired_memory
    assert engine.registry.stats()["entries"] == 0


def test_retired_disk_checkpoint_cleanup_is_retried(engine):
    from shared_runtime.server_memory_checkpoint import pack_memory_state
    c = client(engine)
    first = build(c, [3] * 1024)["memory_embeddings"]
    _, old = pack_memory_state({"memory": first}, c)
    c.checkpoint_saved(old)
    c.release_memory(first)
    second = build(c, [4] * 1024)["memory_embeddings"]
    _, new = pack_memory_state({"memory": second}, c)
    post = c._post
    def fail_delete(route, payload, **kwargs):
        if payload.get("payload", {}).get("operation") == "delete_snapshot":
            raise TimeoutError("snapshot cleanup unavailable")
        return post(route, payload, **kwargs)
    c._post = fail_delete
    c.checkpoint_saved(new)
    assert c._checkpoint_snapshots == new and c._retired_snapshots == old
    c._post = post
    c._delete_retired_snapshots()
    assert not c._retired_snapshots
    engine.registry.resolve(second.descriptor)


def test_failed_child_admission_releases_loaded_handle(engine, tmp_path):
    g, helpers = make_generator(tmp_path, engine, "main")
    memory = build(g.client, [3] * 1024)["memory_embeddings"]
    saved = g.client.save_memory(memory, preserve=True)
    g.server_info = dict(context_length=8, max_total_num_tokens=8, max_req_input_len=8)
    checkpoint = dict(memory=memory, fold_id="rejected", summary_ids=[12], depths=[1]*64, child_id="child")
    with pytest.raises(RuntimeError, match="compaction before admission"):
        g.add_subagent_checkpoint(checkpoint, messages=helpers["history"](), episode_id="main")
    assert not g._folded_branches
    assert engine.registry.stats()["entries"] == 0
    restored = g.client.load_memory(saved)
    assert restored.shape == memory.shape
    g.client.release_memory(restored)


def test_launcher_restores_saved_endpoint_without_environment():
    import runpy
    endpoint = runpy.run_path(str(PAYLOAD.parents[3] / "tools/client_entry.py"))["memory_endpoint"]
    assert endpoint({}, {}) == ""
    assert endpoint({}, {"REMORY_SGLANG_URL": "http://new/"}) == "http://new"
    assert endpoint({"server_memory": {"schema": 1, "endpoint": "http://saved"}},
                    {"REMORY_SGLANG_URL": "http://other"}) == "http://saved"
