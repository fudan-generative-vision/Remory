# Server-owned residual memory

The inference worker holds summary features and soft memory. The client holds
handles and durable snapshot references. Normal encoding, recursive compaction,
parent/child composition and generation never send those tensor values over HTTP.

A handle identifies its backend, server epoch, artifact identity and shape.
Pin a conversation and its children to one backend. A handle is not portable to
another replica; a gateway must not replay a failed stateful request elsewhere.
After a worker restart, restore its persisted snapshot to obtain a fresh handle.
Keep the snapshot volume available to that same backend.

## Connect

Use the advanced serving runtime in this directory with matching model and
compressor artifacts. Its client is separate from the root package's default
SQLite/CPU-memory transport. See [H20 deployment](h20-public-deployment.md) and
[pool setup](inference-pool.md) for installation and routing.

```python
from context_compression.remory_sglang_client import ServerResidualSGLangClient
from shared_runtime.server_memory_checkpoint import (
    save_reference_checkpoint, load_reference_checkpoint,
)

client = ServerResidualSGLangClient(
    "http://127.0.0.1:30280/groups/my-project/backends/h20"
)
client.attest_residual_server("/models/remory-online250")
config = client._residual_config
try:
    result = client.build_residual_memory(
        prefix_ids, history_ids, summary_ids,
        contract=config["summary_contract"],
        memory_budget=config["residual"]["memory_budget"],
    )
    memory = result["memory_embeddings"]  # RemoteMemory descriptor, not a tensor
    save_reference_checkpoint("session/checkpoint.pkl", {
        "memory": memory, "prefix_ids": prefix_ids, "summary_ids": summary_ids,
        "slot_depths": result["receipt"]["slot_depths"],
    }, client)
finally:
    client.close_memory()  # durable checkpoint snapshots remain available
```

Set `PYTHONPATH` to `src/recursive_memory/stack/payload/workspace` when using
these modules from a checkout. The environment needs the pinned client runtime
and local tokenizer/configuration files; creating a client does not start or
replace a model server. A gateway holds upstream credentials; clients of a
loopback gateway do not need to receive them.

On resume, attest the same backend, call `load_reference_checkpoint`, then save
the restored state before the next generation. `generate_with_memory` takes the
returned `RemoteMemory` as `memory_embeddings` and exact memory placeholder
positions. `build_residual_memory` accepts that same object as `source_memory`
for recursive compaction. See the tests for complete small protocol examples.

Only load private checkpoints you created. Checkpoints include conversation
state, use Python serialization, and are not an exchange format for untrusted
files. The helpers require private ownership and the original backend.

## Budget and commit order

The request's memory budget is preserved. Omitting it uses the selected
checkpoint policy (4096 slots in the online-250 deployment). An explicit
`memory_budget=None` allows an unbounded planner; model context capacity and
server quotas still apply. The transport must never silently override this
choice. Block geometry remains 1024 source positions to 64 soft positions.

The caller first publishes the new checkpoint atomically and fsyncs it. Only
then may it accept collected samples, close the organizer and retire the old
memory. `save_reference_checkpoint` commits snapshot ownership after the file
is durable. A caller using `ResidualGenerator` then invokes
`commit_remote_retirements()`. A failed write leaves the previous checkpoint
and its roots available. Do not treat an uncommitted organizer result as evidence.

Child checkpoints carry owner/model identity, backend, summary, depths and a
snapshot. Reject mismatched ownership or invalid geometry before admission.
Repeated child admission must not consume the same branch twice. Rejected
imports must not delete a child's persisted snapshot.

## Existing CPU checkpoints

Preserve the original checkpoint. `restore_memory_state(state, client)` can
import legacy CPU tensors once, then `save_reference_checkpoint` rewrites the
state as references. This one-time import transfers tensor values; normal
requests after migration do not. Do not mix a remote backend with an unsupported
local encoder-version registry selection.

Active clients renew leases every minute or sooner when TTL requires it.
Abandoned handles expire. Snapshots referenced by a committed checkpoint survive
client shutdown. A transient renewal error is not permission to move the handle
to another backend; retain the checkpoint and diagnose the pinned service.

Remote capture can retain snapshot references, but the current local trainer
expects CPU tensor samples. Those remote samples must be excluded from that
trainer until it has a reader beside the inference host. No automatic online
training or weight migration is implied by adding a remote backend.
