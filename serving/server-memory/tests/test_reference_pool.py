import asyncio
import json
import pytest
from recursive_memory.stack.request_queue import Backend,AdmissionQueue
from recursive_memory.stack.gateway import outbound_request

def test_pinned_only_memory_and_plain_offload():
 async def check():
  local=Backend('local','http://local','m',4,healthy=True,observed_load=4)
  remote=Backend('h20','http://remote','m',2,healthy=True,memory_pinned_only=True)
  q=AdmissionQueue([local,remote])
  memory=q.submit(group='legacy',model='m',requires_full_backend=True)
  plain=q.submit(group='plain',model='m');assert plain.backend is remote
  pinned=q.submit(group='pinned',model='m',pinned='h20',requires_full_backend=True);assert pinned.backend is remote
  assert memory.backend is None
  q.release(plain);remote.observed_starts=remote.starts;remote.observed_load=1
  q.pump();assert memory.backend is None
  local.observed_load=0;q.pump();assert memory.backend is local
  for t in [pinned,memory]:q.release(t)
 asyncio.run(check())

def test_pinned_h20_forbids_normal_tensor_transport():
 b=Backend('h20','http://remote','m',2,memory_pinned_only=True)
 data={'sampling_params':{'custom_params':{'specforge_context_compressor':{'mode':'recover','memory_embeddings':[[0.]]}}}}
 with pytest.raises(ValueError,match='references'):
  outbound_request(data,json.dumps(data).encode(),False,'/generate',b,False)
 # Explicit one-time import is the separate, permitted operation.
 data={'payload':{'operation':'import','values':[[0.]]}}
 body,_=outbound_request(data,json.dumps(data).encode(),False,'/remory/memory',b,False)
 assert json.loads(body)==data
