import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import os
if not os.environ.get("STACK_INTEGRATION"):
    pytest.skip("requires controller dependencies", allow_module_level=True)
import torch

from shared_runtime.residual_backend import ResidualGenerator
from shared_runtime.subagent_memory import export_checkpoint, read_checkpoint
from terminal_bench_eval.codex_r2eact_gateway import R2EActGatewayConfig, _supported_tools_and_routes
from context_compression.grepseek_sglang_client import SGLangGeneration


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0
    def encode(self, text, **kwargs):
        return [ord(c) + 1000 for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i - 1000) for i in ids if i >= 1000)


class Generator(ResidualGenerator):
    def _canonical(self, messages, episode_id):
        return copy.deepcopy(messages)
    def _chat(self, messages, *, opening=False):
        text = ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages)
        return self.tokenizer.encode(text + ('<assistant>' if opening else ''))


class Client:
    def __init__(self):
        self.calls, self.encodes = [], []
        self.fail = False
    def generate_discrete(self, ids, **kwargs):
        self.calls.append((list(ids), kwargs))
        text = json.dumps(dict(current_progress=['Child found CEDAR-719.'], key_decisions=[],
            important_context_constraints_preferences=['Return evidence to parent.'], next_steps=[],
            critical_data_examples_references=['file result.txt']))
        return SGLangGeneration(tuple(Tokenizer().encode(text)) + (1,), True,
                                {'type': 'stop', 'matched': 1}, {})
    def generate_with_memory(self, ids, **kwargs):
        return self.generate_discrete(ids, **kwargs)
    def build_residual_memory(self, prefix, history, body, **kwargs):
        if self.fail:
            raise RuntimeError('encoder unavailable')
        self.encodes.append((list(prefix), list(history), list(body), kwargs))
        depth = max(kwargs.get('source_memory_depths', [0]), default=0) + 1
        return dict(memory_embeddings=torch.full((64, 2), float(depth)), receipt={'slot_depths': [depth]*64})


def make(tmp_path, name):
    deployment = dict(summary_contract=dict(prompt='Summarize.', temperature=0.0, end_token_id=1,
        student_summary_before_ids=[801], student_summary_after_ids=[802],
        student_memory_before_ids=[803], student_memory_after_ids=[804]),
        residual=dict(block_tokens=1024, memory_budget=4096, compression_ratio=16),
        compressor=dict(max_depth=8, target_hidden_size=2))
    cfg = R2EActGatewayConfig(specforge_root=tmp_path, checkpoint=tmp_path, corpus_dir=tmp_path / name)
    client = Client()
    generator = Generator(Tokenizer(), client, dict(context_length=262144,
        max_total_num_tokens=524288, max_req_input_len=262143), cfg, [], deployment)
    generator.episode_id = name
    return generator, client


def history():
    return [{'role': 'system', 'content': 'SYSTEM'}, {'role': 'user', 'content': 'ROOT-TASK'},
            {'role': 'assistant', 'content': 'READ-FILE'}, {'role': 'tool', 'content': 'CEDAR-719'}]


def checkpoint(tmp_path, name='child', parent='parent'):
    generator, client = make(tmp_path, name)
    messages = history() + [{'role': 'assistant', 'content': 'FINAL-EVIDENCE'}]
    session = SimpleNamespace(messages=messages, episode_id=name, raw_suffix_images=[])
    path = export_checkpoint(SimpleNamespace(_generator=generator), session, tmp_path / name,
                             child_id=name, parent_id=parent)
    assert 'FINAL-EVIDENCE' in generator.tokenizer.decode(client.encodes[-1][1])
    assert session.messages == messages
    return path


def test_short_child_folds_immediately_and_preserves_parent_raw_history(tmp_path):
    path = checkpoint(tmp_path)
    parent, client = make(tmp_path, 'parent')
    messages = history()
    _, prefix, closed = parent._text_history(messages, 'parent')
    item = read_checkpoint(path, parent, parent_id='parent', child_id='child')
    assert parent.add_subagent_checkpoint(item, messages=messages, episode_id='parent')
    assert not parent.add_subagent_checkpoint(item, messages=messages, episode_id='parent')
    ids, images, reserve = parent._render(messages, episode_id='parent')
    assert ids[:len(closed)] == closed
    assert parent.summary_ids is None and parent.consumed_rounds == 0
    assert parent._context_memory.shape == (64, 2)
    assert all(ids[p] == 0 for p in parent.memory_positions)
    parent._decode(ids, images, reserve)
    assert torch.equal(client.calls[-1][1]['memory_embeddings'], item['memory'])
    assert 'CEDAR-719' in parent.tokenizer.decode(ids)


def test_two_children_survive_recursive_parent_compaction(tmp_path):
    parent, client = make(tmp_path, 'parent')
    messages = history()
    for name in ('a', 'b'):
        item = read_checkpoint(checkpoint(tmp_path, name), parent, parent_id='parent', child_id=name)
        parent.add_subagent_checkpoint(item, messages=messages, episode_id='parent')
        messages += [{'role':'assistant','content':'PARENT-CONTINUES'}, {'role':'tool','content':name}]
    old = copy.deepcopy(parent._folded_branches)
    summary = parent.summarize(messages=messages, episode_id='parent', round_index=len(messages)//2)
    encode = client.encodes[-1]
    assert encode[3]['source_memory'].shape == (128, 2)
    assert len(encode[3]['source_memory_positions']) == 128
    assert encode[3]['source_memory_depths'] == [1]*128
    assert summary['previous_memory_slots'] == 128
    assert len(parent._folded_branches) == 2
    parent.install(episode_id='parent', target_completed_rounds=len(messages)//2-1, summary=summary)
    assert parent._folded_branches == []
    assert parent.memory_depths == [2]*64
    for item in old:
        assert not parent.add_subagent_checkpoint(item, messages=messages, episode_id='parent')
    messages += [{'role':'assistant','content':'NEW'}, {'role':'tool','content':'NEW-RESULT'}]
    parent.summarize(messages=messages, episode_id='parent', round_index=len(messages)//2)
    assert client.encodes[-1][3]['source_memory_depths'] == [2]*64


def test_failed_compaction_keeps_child_memory_and_original_frontier(tmp_path):
    parent, client = make(tmp_path, 'parent')
    item = read_checkpoint(checkpoint(tmp_path), parent, parent_id='parent', child_id='child')
    parent.add_subagent_checkpoint(item, messages=history(), episode_id='parent')
    client.fail = True
    with pytest.raises(RuntimeError, match='encoder unavailable'):
        parent.summarize(messages=history(), episode_id='parent', round_index=2)
    assert parent.consumed_ids is None
    assert len(parent._folded_branches) == 1
    assert parent._prepared is None


def test_checkpoint_rejects_other_owner_model_and_corruption(tmp_path):
    path = Path(checkpoint(tmp_path))
    parent, _ = make(tmp_path, 'parent')
    with pytest.raises(RuntimeError, match='owner/model'):
        read_checkpoint(path, parent, parent_id='other', child_id='child')
    parent.deployment['compressor']['target_hidden_size'] = 3
    with pytest.raises(RuntimeError, match='owner/model'):
        read_checkpoint(path, parent, parent_id='parent', child_id='child')
    parent.deployment['compressor']['target_hidden_size'] = 2
    path.with_name('memory.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='checksum'):
        read_checkpoint(path, parent, parent_id='parent', child_id='child')


def test_parent_with_existing_memory_keeps_both_memories(tmp_path):
    parent, client = make(tmp_path, 'parent')
    messages = history()
    summary = parent.summarize(messages=messages, episode_id='parent', round_index=2)
    parent.install(episode_id='parent', target_completed_rounds=1, summary=summary)
    messages += [{'role':'assistant','content':'SPAWN'}, {'role':'tool','content':'child'}]
    item = read_checkpoint(checkpoint(tmp_path), parent, parent_id='parent', child_id='child')
    parent.add_subagent_checkpoint(item, messages=messages, episode_id='parent')
    ids, _, _ = parent._render(messages, episode_id='parent')
    assert len(parent.memory_positions) == 128
    assert parent.memory_positions == sorted(set(parent.memory_positions))
    assert parent._context_memory.shape == (128, 2)
    assert 'SPAWN' in parent.tokenizer.decode(ids)


def test_native_namespace_routes_only_supported_delegation_tools():
    def tool(name):
        return dict(type='function', name=name, description=name,
                    parameters={'type':'object','properties':{}})
    request = {'tools':[tool('exec_command'), dict(type='namespace', name='multi_agent_v1',
               tools=[tool(name) for name in ('spawn_agent','wait_agent','close_agent','send_input','resume_agent')])]}
    tools, routes = _supported_tools_and_routes(request, subagents_enabled=True)
    assert [t['name'] for t in tools] == ['exec_command','spawn_agent','wait_agent','close_agent']
    assert routes['spawn_agent'].namespace == 'multi_agent_v1'
    assert tools[1]['parameters']['required'] == ['message']
    assert set(tools[1]['parameters']['properties']) == {'message'}
    disabled, routes = _supported_tools_and_routes(request)
    assert [t['name'] for t in disabled] == ['exec_command']
    assert routes == {}


def test_async_ready_checkpoint_enters_only_at_parent_boundary(tmp_path):
    from shared_runtime.subagents import SubagentGateway
    parent, _ = make(tmp_path, 'parent')
    backend = SimpleNamespace(config=parent.config, serving_concurrency=4, _generator=parent)
    router = SubagentGateway(backend, tmp_path / 'gateway')
    router.root_id = 'parent'
    child = SimpleNamespace(checkpoint=None)
    router.children['child'] = child
    session = SimpleNamespace(messages=history(), episode_id='parent', round_index=2, raw_suffix_images=[])
    router._fold_completed(session)
    assert parent._folded_branches == []
    child.checkpoint = checkpoint(tmp_path)
    assert parent._folded_branches == []
    assert router._validate_final(session) is not None
    router._fold_completed(session)
    assert len(parent._folded_branches) == 1
    assert router._validate_final(session) is None
    router._fold_completed(session)
    assert len(parent._folded_branches) == 1


def test_failed_fold_admission_is_atomic(tmp_path):
    parent, _ = make(tmp_path, 'parent')
    item = read_checkpoint(checkpoint(tmp_path), parent, parent_id='parent', child_id='child')
    _, prefix, closed = parent._text_history(history(), 'parent')
    parent.server_info['context_length'] = len(closed) + parent.config.capacity_pressure_reserve_tokens + 1
    with pytest.raises(RuntimeError, match='requires parent compaction'):
        parent.add_subagent_checkpoint(item, messages=history(), episode_id='parent')
    assert parent._folded_branches == []
    assert parent._folded_ids == set()
    assert parent._context_memory is None


def test_child_archive_recall_cannot_target_parent_or_sibling(tmp_path):
    from shared_runtime.subagents import child_memory_config
    parent, _ = make(tmp_path, 'parent')
    backend = SimpleNamespace(config=parent.config, memory_mirror_path=tmp_path / 'memory.jsonl')
    first, first_mirror = child_memory_config(backend, 'first')
    second, second_mirror = child_memory_config(backend, 'second')
    assert len({backend.config.memory_guest_path, first.memory_guest_path, second.memory_guest_path}) == 3
    assert first_mirror.parent == second_mirror.parent == backend.memory_mirror_path.parent
    assert first_mirror.name == Path(first.memory_guest_path).name
    assert first_mirror != second_mirror != backend.memory_mirror_path
