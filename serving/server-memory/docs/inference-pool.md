# Mixed inference pool and reference-only memory

This update adds two encoder optimizations and durable remote-memory transport.
It also fixes native organizers that remain alive after a runtime thread panic.
The root quick start remains unchanged; this guide is for the advanced serving
runtime in `serving/server-memory`.

## Add a remote replica

[The example config](../gateway.reference.example.json) combines four local
replicas at four requests each with one remote TP4 replica at two requests: 18
slots in total. Use capacities measured on your engines. Edit the model paths,
backend URL and credential-file path before starting the gateway.

```bash
cd serving/server-memory
python tools/add_remote_backend.py \
  --config gateway.example.json \
  --base-url https://h20.example.org/backends/h20-tp4 \
  --api-key-file /secure/h20.key \
  --local-checkpoint /models/remory-online250 \
  --max-inflight 2 --memory-pinned-only \
  --output gateway.with-h20.json
PYTHONPATH="$PWD/src" python -m recursive_memory.stack.gateway \
  --config gateway.with-h20.json --host 127.0.0.1 --port 30280
```

This uses the native SGLang backend URL, without a trailing `/v1`. The pool owns
the upstream credential. Ordinary generation can use any compatible replica.
A remote entry with `memory_pinned_only: true` accepts memory operations only
when explicitly selected, for example:

```text
http://127.0.0.1:30280/groups/my-project/backends/h20
```

Save this route with the conversation. Use it for the parent, children, compact,
checkpoint restore and generation. The caller opts in with
`REMORY_SERVER_MEMORY=1`; `REMORY_SERVING_CONCURRENCY` may select the validated
per-client capacity. Never advertise a larger capacity than the engine supports.
The checkpoint helpers and lifecycle are described in [server memory](server-memory.md).

Keep `allow_tensor_recover: false` for the reference-only route. Normal generation
rejects hidden-state/soft-token arrays on that backend. An explicit legacy
checkpoint import is the one allowed tensor transfer; subsequent checkpoints
store snapshots. Backend identity and local encoder-version support remain
routing requirements. Sessions using a local version registry stay local until
the remote engine supports their selected version.

For an existing pool, stop new requests, preserve checkpoints and drain in-flight
requests before changing the gateway/engines. Restore the same sessions and goals
after endpoint tests. Do not replay a stateful POST after an ambiguous failure.
This repository does not start, pause or move another operator's research sessions.

## Artifact identity

Prefer the same public checkpoint on both hosts. Raw weight-file and serving
contract hashes must match. One exception is available for the exact online-250
`.pt` and safetensors pair in [this verification record](online250-tensor-equivalence.json):
117 tensors match in metadata and value hashes. The exception is opt-in, fixed to
that pair and hook ABI, and does not rewrite either server's raw identity.

For that specific pair only, add
`--verified-artifact-pair h20_online250_verified_tensor_pair_20261109_v1` when
building the config, and set these variables for the gateway and caller:

```bash
export REMORY_H20_VERIFIED_PAIR=h20_online250_verified_tensor_pair_20261109_v1
export REMORY_H20_EQUIVALENCE_PROOF="$PWD/docs/online250-tensor-equivalence.json"
```

This is not a general mechanism for declaring two models equivalent. A changed
proof, geometry, hash or hook ABI fails validation.

## Encoder execution

Single-device inference batches adjacent blocks with the same recursion depth.
Each block keeps its own positions, mask and 1024-to-64 geometry. Summary K/V
projections are shared only during inference for an identical expanded summary.
Training and distinct summaries keep their existing paths.

```bash
export REMORY_ENCODE_BATCH_BLOCKS=4
export REMORY_ENCODE_BATCH_MIB=1024
```

The encoder estimates free memory, reserves headroom and reduces the batch after
a CUDA allocation failure. Set `REMORY_ENCODE_BATCH_BLOCKS=1` for serial local
encoding. Successful blocks are not recomputed during this fallback.

For TP inference, `REMORY_PARALLEL_ENCODE=1` distributes independent nodes across
the actor's TP group. Each node uses a serial compressor forward; ranks gather
results in original block order. The full causal actor prefix is unchanged.
With TP parallel encoding disabled, workers use the local block path. Disable
both optimizations for the original serial behavior:

```bash
export REMORY_PARALLEL_ENCODE=0
export REMORY_ENCODE_BATCH_BLOCKS=1
```

Encoding responses include a `remory_encoding` timing receipt, including for
zero-output-token prefill requests. The optimized code must be loaded by a new
engine process; changing an environment variable does not modify a running worker.

## Measurements and validation

[Recorded measurements](performance-20261109.json) distinguish compressor timing
from actor prefill, summary generation, network transfer and queueing:

| Path | Measured compressor speedup | Numerical result |
| --- | --- | --- |
| Local block batches | 1.09–1.67× | BF16 normalized RMS difference ≤0.56%; minimum slot cosine >0.99994 |
| H20 TP4 independent nodes, multiple blocks | 2.40–3.23× | Bitwise equal to serial in the tested cases |

A single-node H20 case measured 0.97×. These are component benchmarks, not an
end-to-end research throughput or downstream-quality claim. Local and remote
GPU measurements were collected from their respective deployed implementations;
the combined dispatch in this public snapshot is additionally covered by CPU tests.

The public snapshot passed 261 distinct regression tests across routing,
encoding, checkpoint/child lifecycle, packaging and native Codex with a fake
inference service. Optional skips were rerun with their dependencies enabled.

Live synthetic checks covered recursive memory, child composition, durable
resume, native agentic compaction and post-upgrade encode/generate. The deployed
pool restored 19 existing research sessions with their original thread/goal/run
identities. A short health-probe timeout and a lease-renewal 503 occurred under
load and recovered; they are not omitted from the deployment record. No research
text, endpoints, credentials or session journals are included here.

Remote capture currently records snapshots and is excluded from the local
CPU-tensor online trainer. A trainer beside the inference host needs an explicit
snapshot reader before those samples can participate in training.
