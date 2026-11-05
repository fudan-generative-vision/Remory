from dataclasses import replace
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from remory.models.compressor import SummaryResidualCompressor
from remory.online.learner import FrontierObjective, OnlineEvent
from remory.types import digest


def test_frontier_has_no_future_leakage_and_only_memory_receives_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, attention_dropout=0.0)
    actor = Qwen3ForCausalLM(config).requires_grad_(False).eval()
    target = SimpleNamespace(model=actor, language_model=actor.model, lm_head=actor.lm_head,
        device=torch.device('cpu'), hidden_size=16, target_layer_ids=(0, 1),
        embed_ids=lambda ids: actor.get_input_embeddings()(torch.tensor([ids])),
        student_hidden=lambda inputs, grad: actor.model(inputs_embeds=inputs, use_cache=False).last_hidden_state)
    draft = Qwen3Config(vocab_size=100, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, layer_types=['sliding_attention', 'full_attention'], sliding_window=8)
    compressor = SummaryResidualCompressor(draft, target_hidden_size=16, target_layer_ids=(0, 1),
        compression_ratio=2, local_window=2048, gradient_checkpointing=False, max_depth=4)
    contract = dict(student_summary_before_ids=[10], student_summary_after_ids=[11],
        student_memory_before_ids=[12], student_memory_after_ids=[13])
    objective = FrontierObjective(target, compressor, contract, memory_budget=4,
        block=8, ratio=2, max_depth=4, projection_tokens=2)
    event = OnlineEvent(key='tiny', turn=0, prefix_ids=(1, 2), history_ids=tuple(range(20, 34)),
        opening_ids=(14,), target_ids=(40, 41, 42), source_memory=torch.randn(4, 16),
        source_memory_positions=(4, 5, 12, 13), source_memory_depths=(1, 1, 1, 1))
    features, teacher = objective.raw_teacher_and_source(event)
    other_features, other_teacher = objective.raw_teacher_and_source(replace(event, target_ids=(50, 51, 52)))
    torch.testing.assert_close(features, other_features, rtol=0, atol=0)
    assert not torch.equal(teacher, other_teacher)
    summary = {'ids': [60, 61], 'summary_sha256': digest([60, 61])}
    before = {k: v.detach().clone() for k, v in compressor.named_parameters()}
    metrics = objective.backward_response(event, summary)
    assert metrics['tokens'] == 3 and metrics['covered_source_tokens'] == len(event.history_ids)
    assert all(p.grad is None and not p.requires_grad for p in actor.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0 for p in compressor.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in compressor.parameters())
    torch.optim.AdamW(compressor.parameters(), lr=1e-4).step()
    assert any(not torch.equal(before[k], v) for k, v in compressor.named_parameters())
