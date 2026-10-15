import copy
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config
from context_compression.grepseek_modeling import GrepSeekDFlashCompressor
from context_compression.qwen38_dflash2_compressor import (
    DFlash2Compressor, PaddedGroupedConv, load_pretrained_backbone, SELECTOR_KEYS,
)


def tiny(checkpointing=False):
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        layer_types=['sliding_attention']*2+['full_attention'],
                        sliding_window=32, attention_dropout=0.0)
    config.dflash_config = dict(block_size=8, conv_kernel_size=2, conv_group_size=4,
                               target_layer_ids=[1, 3])
    return DFlash2Compressor(config, target_hidden_size=32, target_layer_ids=(1, 3),
                            compression_ratio=4, local_window=32,
                            gradient_checkpointing=checkpointing)


def inputs():
    return torch.randn(2, 64, 64), torch.ones(2, 64, dtype=torch.bool)


def test_identity_convolutions_preserve_original_compressor():
    torch.manual_seed(4)
    model = tiny()
    old = GrepSeekDFlashCompressor(model.config, target_hidden_size=32, target_layer_ids=(1,3),
                                  compression_ratio=4, local_window=32, gradient_checkpointing=False)
    old.load_state_dict({k:v for k,v in model.state_dict().items() if '_conv.' not in k})
    x, mask = inputs()
    torch.testing.assert_close(model(x, mask).embeddings, old(x, mask).embeddings, rtol=0, atol=0)


def test_pretrained_load_is_exact_and_only_selector_is_excluded(tmp_path):
    source, destination = tiny(), tiny()
    state = {k:v for k,v in source.state_dict().items() if k not in ('mask_embedding','output_embedding_rms')}
    state.update({name: torch.randn(2, 3) for name in SELECTOR_KEYS})
    path = tmp_path/'model.safetensors'
    save_file(state, path)
    receipt = load_pretrained_backbone(destination, path)
    for name, value in state.items():
        if name not in SELECTOR_KEYS:
            torch.testing.assert_close(destination.state_dict()[name], value, rtol=0, atol=0)
    assert receipt['loaded_parameters'] == sum(v.numel() for k,v in state.items() if k not in SELECTOR_KEYS)
    del state['fc.weight']
    save_file(state, path)
    with pytest.raises(ValueError, match='keys differ'):
        load_pretrained_backbone(destination, path)


def test_checkpoint_recomputation_preserves_convolution_gradients():
    torch.manual_seed(8)
    a = tiny(False)
    with torch.no_grad():
        for name, parameter in a.named_parameters():
            if '_conv.kernel_projection.weight' in name:
                parameter.normal_(0, 0.003)
    b = copy.deepcopy(a)
    b.gradient_checkpointing = True
    x, mask = inputs()
    for model in (a,b):
        out = model(x, mask).embeddings
        (out * torch.linspace(-1,1,out.numel()).reshape_as(out)).sum().backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7)
        if '_conv.' in name:
            assert p.grad.norm() > 0, name


def test_partial_convolution_block_has_no_future_or_cross_example_leakage():
    torch.manual_seed(3)
    conv = PaddedGroupedConv(32,8,2,4)
    with torch.no_grad():
        conv.kernel_projection.weight.normal_(0,0.005)
        conv.base_kernel[:,1].fill_(0.1)
    x = torch.randn(2,11,32)
    out, kernel = conv.prepare(x)
    actual = conv.finish(out,kernel)
    changed = x.clone()
    changed[0,8:] += 100
    changed[1] += 50
    altered, k = conv.prepare(changed)
    altered = conv.finish(altered,k)
    torch.testing.assert_close(actual[0,:8],altered[0,:8],rtol=0,atol=0)
    assert actual.shape == x.shape and torch.isfinite(actual).all()


def test_padded_source_tokens_do_not_change_memory():
    model = tiny()
    model.local_window = 2048  # The production padded-slot contract.
    x, mask = inputs()
    mask[0,20:] = False
    lengths = torch.tensor([20,64])
    slots = torch.tensor([64,64])
    first = model(x,mask,source_token_lengths=lengths,slot_token_lengths=slots).embeddings
    x[0,20:] = torch.randn_like(x[0,20:])*1000
    second = model(x,mask,source_token_lengths=lengths,slot_token_lengths=slots).embeddings
    torch.testing.assert_close(first,second,rtol=0,atol=0)
