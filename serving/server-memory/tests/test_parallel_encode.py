import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/recursive_memory/stack/payload/workspace'))
from context_compression.remory_parallel_encode import encode_blocks


class FakeCompressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.fail = False
        self.calls = 0

    def forward(self, features, mask, *, source_token_lengths, slot_token_lengths,
                summary_features, summary_attention_mask, input_depth):
        self.calls += 1
        if self.fail and input_depth == 3:
            raise ValueError('injected rank failure')
        # Reflect row order, real partial length, fixed slot geometry and depth.
        value = features.sum(1, keepdim=True) + input_depth * 10000
        value = value + source_token_lengths[:, None, None]
        embeddings = value.expand(-1, int(slot_token_lengths[0])//2, -1).clone()
        return SimpleNamespace(embeddings=embeddings, attention_mask=torch.ones(embeddings.shape[:2],dtype=torch.bool))


def worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4)
    try:
        model = FakeCompressor()
        for length in [1, 8, 9, 17, 31, 32, 33, 69]:
            aux = [torch.arange((length+3)*2,dtype=torch.float32).reshape(length+3,2)]
            kwargs = dict(start=3,end=3+length,block_tokens=8,compression_ratio=2,
                          hidden_size=2,summary=torch.ones(1,3,2),summary_mask=torch.ones(1,3,dtype=torch.bool),
                          depths=[i%4 for i in range((length+7)//8)])
            with torch.inference_mode():
                expected=encode_blocks(model,aux,parallel=False,**kwargs)
                model.calls=0
                actual=encode_blocks(model,aux,parallel=True,**kwargs)
            assert torch.equal(actual,expected), (rank,length)
            count=len(kwargs['depths'])
            assert model.calls == (count if count==1 else len(range(rank,count,4)))
        model.fail=True
        kwargs.update(end=35,depths=[0,1,2,3])
        try:
            encode_blocks(model,[torch.ones(35,2)],parallel=True,**kwargs)
        except RuntimeError as exc:
            assert 'injected rank failure' in str(exc)
        else:
            raise AssertionError('failure must propagate to every rank')
    finally:
        dist.destroy_process_group()


def test_tp_order_partial_depth_and_collective_failure(tmp_path):
    mp.spawn(worker,args=('file://'+str(tmp_path/'rendezvous'),),nprocs=4,join=True)
