"""Numerical and failure-path checks for serving-only residual microbatching."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from transformers import Qwen3Config

PAYLOAD = Path(__file__).resolve().parents[1] / "src/recursive_memory/stack/payload"
sys.path.insert(0, str(PAYLOAD / "workspace"))
from context_compression.qwen38_residual_sglang_hook import encode_residual_blocks
from context_compression.qwen38_residual_modeling import SummaryResidualCompressor
from context_compression.qwen38_dflash2_residual import DFlash2SummaryResidualCompressor


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_model(kind):
    torch.manual_seed(71)
    config = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=8192, attention_dropout=0.0)
    config.dflash_config = dict(block_size=8, conv_kernel_size=2, conv_group_size=4)
    cls = DFlash2SummaryResidualCompressor if kind == "dflash2" else SummaryResidualCompressor
    model = cls(config, target_hidden_size=16, target_layer_ids=(0, 1),
        compression_ratio=16, local_window=2048, gradient_checkpointing=False).eval()
    # Exercise learned depth, summary and dynamic-convolution effects, not only
    # the zero/identity initialization of these branches.
    with torch.no_grad():
        model.depth_embedding.weight.normal_(std=0.2)
        for cross in model.summary_attention:
            cross.gate.fill_(0.4)
        for name, parameter in model.named_parameters():
            if "kernel_projection.weight" in name:
                parameter.normal_(std=0.03)
    return model


def old_serial(model, aux, start, end, summary, summary_mask, depths):
    pieces = []
    with torch.inference_mode():
        for i, left in enumerate(range(start, end, 1024)):
            right = min(end, left + 1024)
            features = torch.cat([h[left:right] for h in aux], -1)[None]
            encoded = model(features, torch.ones(features.shape[:2], dtype=torch.bool),
                source_token_lengths=torch.tensor([right - left]), slot_token_lengths=torch.tensor([1024]),
                summary_features=summary, summary_attention_mask=summary_mask, input_depth=depths[i])
            pieces.append(encoded.embeddings[encoded.attention_mask])
    return torch.cat(pieces)


@pytest.mark.parametrize("kind", ["residual", "dflash2"])
@pytest.mark.parametrize("summary_rows", [19, 4097])
@pytest.mark.parametrize("depths", [[0, 0, 0, 0], [1, 1, 0, 0]])
def test_matches_original_serial_with_partial_tail_and_summary_streaming(kind, summary_rows, depths):
    model = tiny_model(kind)
    start, end = 7, 7 + 3 * 1024 + 17
    aux = [torch.randn(end, 16), torch.randn(end, 16)]
    summary = torch.randn(1, summary_rows, 16)
    summary_mask = torch.ones(1, summary_rows, dtype=torch.bool)
    summary_mask[:, 2:5] = False
    expected = old_serial(model, aux, start, end, summary, summary_mask, depths)
    actual = encode_residual_blocks(model, aux, start=start, end=end, block=1024, ratio=16,
        summary=summary, summary_mask=summary_mask, input_depths=depths, max_batch_blocks=4)
    assert actual.shape == (256, 16)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)


class InstrumentedCompressor(torch.nn.Module):
    def __init__(self, *, oom_above=None, error=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = []
        self.oom_above, self.error = oom_above, error

    def forward(self, features, mask, *, input_depth, **kwargs):
        batch = features.shape[0]
        self.calls.append((batch, input_depth))
        if self.error is not None:
            raise self.error
        if self.oom_above is not None and batch > self.oom_above:
            raise torch.cuda.OutOfMemoryError("injected allocation failure")
        value = (features * mask[..., None]).sum(1) / mask.sum(1)[:, None] + input_depth
        return SimpleNamespace(embeddings=value[:, None].expand(-1, 64, -1),
            attention_mask=torch.ones(batch, 64, dtype=torch.bool))


def instrumented_run(model, depths, **kwargs):
    aux = [torch.arange(len(depths) * 1024 - 19, dtype=torch.float32)[:, None]]
    return encode_residual_blocks(model, aux, start=0, end=len(aux[0]), block=1024, ratio=16,
        summary=torch.zeros(1, 4, 1), summary_mask=torch.ones(1, 4, dtype=torch.bool),
        input_depths=depths, **kwargs)


def test_oom_halves_only_failed_batch_and_does_not_repeat_completed_nodes():
    model = InstrumentedCompressor(oom_above=2)
    depths = [1, 1, 0, 0, 0, 0]
    actual = instrumented_run(model, depths, max_batch_blocks=4)
    expected = instrumented_run(InstrumentedCompressor(), depths, max_batch_blocks=1)
    torch.testing.assert_close(actual, expected)
    assert model.calls == [(2, 1), (4, 0), (2, 0), (2, 0)]


def test_single_block_oom_and_other_errors_propagate():
    for error in (torch.cuda.OutOfMemoryError("single block"), RuntimeError("model bug")):
        model = InstrumentedCompressor(error=error)
        with pytest.raises(type(error), match=str(error)):
            instrumented_run(model, [0], max_batch_blocks=1)
        assert len(model.calls) == 1


def test_environment_controls_batching_and_serial_escape_hatch(monkeypatch):
    for limit, expected in [(4, [(4, 0), (1, 0)]), (1, [(1, 0)] * 5)]:
        monkeypatch.setenv("REMORY_ENCODE_BATCH_BLOCKS", str(limit))
        model = InstrumentedCompressor()
        instrumented_run(model, [0] * 5)
        assert model.calls == expected


def test_memory_headroom_limits_cuda_batch(monkeypatch):
    from context_compression.qwen38_residual_sglang_hook import _residual_microbatch_limit
    weight = SimpleNamespace(device=torch.device("cuda:0"), element_size=lambda: 2)
    model = SimpleNamespace(parameters=lambda: iter([weight]))
    monkeypatch.setenv("REMORY_ENCODE_BATCH_BLOCKS", "4")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (512 << 20, 4 << 30))
    assert _residual_microbatch_limit(model, [torch.zeros(1, 16)], torch.zeros(1, 20, 16), 1024, 16) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_shared_summary_projects_once_without_mixing_query_masks(stream):
    cross = tiny_model("dflash2").summary_attention[0]
    slots = torch.randn(3, 8, 16)
    summary = torch.randn(1, 19, 16).expand(3, -1, -1)
    mask = torch.ones(3, 19, dtype=torch.bool)
    mask[1, :5] = False
    calls = []
    hook = cross.k_proj.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    with torch.inference_mode():
        actual = cross(slots, summary, mask, stream_summary=stream, chunk_tokens=7)
        shared_calls = list(calls)
        expected = cross(slots, summary.clone(), mask, stream_summary=stream, chunk_tokens=7)
    hook.remove()
    assert shared_calls == ([1, 1, 1] if stream else [1])
    assert calls[len(shared_calls):] == ([3, 3, 3] if stream else [3])
    torch.testing.assert_close(actual, expected)
