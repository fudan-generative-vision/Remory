import sys
from types import SimpleNamespace
import pytest
import torch
from test_remory_server_memory import engine, client, build


@pytest.mark.parametrize('enabled',[True,False])
def test_hook_selects_only_the_actor_tp_group(engine,monkeypatch,enabled):
    import context_compression.remory_parallel_encode as parallel
    group=object();calls=[]
    monkeypatch.setenv('REMORY_PARALLEL_ENCODE','1' if enabled else '0')
    monkeypatch.setitem(sys.modules,'sglang.srt.distributed',SimpleNamespace(get_tp_group=lambda:SimpleNamespace(device_group=group)))
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda g:2 if g is group else (_ for _ in ()).throw(AssertionError('wrong group')))
    def encode(model,aux,**kw):
        calls.append(kw)
        return torch.ones(((kw['end']-kw['start']+1023)//1024*64,2))
    monkeypatch.setattr(parallel,'encode_blocks',encode)
    out=build(client(engine,remote=False),[3]*1030)
    assert out['memory_embeddings'].shape==(128,2)
    assert bool(calls)==enabled
    if enabled:
        assert calls[0]['group'] is group and calls[0]['depths']==[0,0]
