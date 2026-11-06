# Remory memory kept by the inference server

This checkout adds `remory_server_memory_v1` to the Stage-II-14925 architecture
and the pinned online-step-250 continuation. The SGLang worker owns summary features and residual tensors. The
client receives descriptors containing identity, shape and server epoch;
recursive encoding and generation send descriptors back. Tensor composition
also runs on the inference server. Ordinary generation/compaction does not
serialize residual or summary feature values into HTTP requests or responses.

The serving planner has no fixed 4096-slot budget in this mode. It keeps the
existing 1024-to-64 compressor geometry and exact logical depth tracking.
Actual context capacity and the server's memory allocation limits still apply;
an over-capacity request fails without truncating history. Frozen training
metadata and legacy tensor-mode settings retain their original values.

A successful main-session compaction installs the new generation before
releasing the previous main memory, temporary concatenations, and consumed
child handles. Failure keeps the previous roots. Already-admitted requests
hold strong tensor references, so releasing a registry entry does not invalidate
their inputs. A bounded cleanup request is retried by the lease thread after a
transient failure. Active clients renew leases; abandoned handles expire.

## Use the current r-codex

The workspace's `r-codex` launcher now dispatches to this checkout when
`REMORY_SGLANG_URL` is set, or a deployed `deployment.json` selects it for new
sessions. `REMORY_RUNTIME_FILE` selects a verified runtime and checkpoint; the
endpoint must run this patched engine and matching weights. An older server produces an explicit
capability error. Existing sessions with no remote endpoint keep their legacy
runtime.

```bash
REMORY_RUNTIME_FILE="$PWD/deployment-online250/runtime.json" \
  REMORY_SGLANG_URL=http://127.0.0.1:30380/groups/my-project r-codex
# Or use the standalone launcher:
REMORY_SGLANG_URL=http://127.0.0.1:30380/groups/my-project \
  REMORY_RUNTIME_FILE="$PWD/deployment-online250/runtime.json" \
  ./tools/r-codex-server-memory
```

A new session records its endpoint and runtime file in `session.json`. Later,
`r-codex resume SESSION` and `r-codex resume --last` select the saved transport
without requiring the environment variable again. The checkpoint retains the
exact pinned backend route. Parent and native children share that backend;
the request gateway continues to queue their requests by project.

On this execution host, the new harness defaults experimental GPU visibility
to physical GPUs 4–7. GPU 0–3 remain reserved for existing shared inference.
`REMORY_EXPERIMENT_GPUS` can explicitly select a different permitted set when
the client is moved to another execution host. GPU jobs must still use the
shared research queue.

## Start an isolated server from a compatible installed runtime

`tools/serve_server_memory.py` reuses an existing interpreter, model cache and
verified runtime JSON. It copies the installed SGLang Python tree into a
separate directory and applies this checkout's overlay there. It does not
download weights, replace the original engine, or stop another service. The
engine base is the repository's pinned SGLang 0.5.13 recipe. This is a launcher
for the active research runtime, not an installer for arbitrary stock SGLang.
The generic release installer still targets the older step2320 export and
must not be used to substitute that checkpoint for this client's selected release.

Example on a remote host with an allocated four-GPU set and the compatible
runtime already provisioned:

```bash
/path/to/rm-stack/controller/bin/python tools/serve_server_memory.py \
  --state /path/to/runtime.json --runtime-dir /path/to/new-memory-runtime \
  --devices 0,1,2,3 --port 30300 \
  --context 32768 --token-pool 65536 --max-running-requests 2 \
  --memory-fraction 0.75
```

These are initial capacity settings to validate on the target host, not an
H20 throughput measurement. TP size equals the number of selected devices.
Use `--cuda-home` when the default compiler does not support the GPU, and
`--kernel-cache` to select a compatible FlashInfer cache. `--stage-only` stages
the engine without starting inference. The launcher binds inference to loopback.

Serve the queue using **this checkout's** gateway. For a TP4 server, one backend
entry points at its single HTTP port. Independent replicas require separate
entries and independent snapshot directories. Set `max_inflight` no higher
than each engine's `max_running_requests`:

```json
{
  "max_pending": 256,
  "queue_timeout": 1800,
  "request_timeout": 3600,
  "probe_interval": 1,
  "backends": [
    {"name": "remote-tp4", "url": "http://127.0.0.1:30300",
     "model": "Qwen3.8-27B", "max_inflight": 2}
  ]
}
```

```bash
PYTHONPATH="$PWD/src" /path/to/rm-stack/controller/bin/python \
  -m recursive_memory.stack.gateway --config /path/to/gateway.json \
  --host 127.0.0.1 --port 30380
# On the client host; keep this forwarding process running:
ssh -N -L 30380:127.0.0.1:30380 REMOTE_HOST
```

New engines advertise a portable compressor/contract/target metadata identity,
so local and remote model paths can differ. Legacy engines retain their strict
path checks. Both machines must provision the same pinned, unmodified target
weights. The client reads tokenizer/config files from its local cache. An
OpenAI-compatible stock endpoint alone cannot provide this memory protocol.

For a public HTTPS endpoint with API-key authentication and membership in the
local unified pool, use the [H20 deployment handoff](h20-public-deployment.md).
It includes a server-only installer, a verified online-step-250 relocation tool,
a gateway/TLS template, a real protocol smoke test and a local pool config builder.
Client credentials can be supplied through `REMORY_SGLANG_API_KEY_FILE` or
`REMORY_SGLANG_API_KEY`, scoped to the origin in `REMORY_SGLANG_URL`. When using
the unified pool, the upstream key stays in the local gateway's key file.

The gateway advertises compatible memory backends. The client selects one on
initialization and pins subsequent work to it. `/remory/memory` operations join
the normal queue. Lease renewal and handle release use pinned cleanup routes
that remain available when inference capacity is full. References cannot be
silently moved to another replica.

A separately verified H20 replica may also read caller-provided REMORY tensors
without owning the encoder or session. This requires `stateless_only: true`,
`allow_tensor_recover: true`, and an exact `tensor_recover_artifact_identity`
pin, in addition to the target identity anchor and required capabilities.
Only self-contained, cache-bypassed `/generate` recovery with explicit tensor
rows/positions and the required custom logit processor qualifies. Images and
SSE can accompany that request; batched or stateful recovery does not qualify.
The gateway strips the local encoder-version marker only from this verified
remote recover dispatch. Encoding, local training-version selection, handles
and session state remain local. A pin change quarantines the replica, and queued
requests recheck identity and capabilities before dispatch.

The [sanitized verification record](tensor-recovery-verification.json) includes
a target-embedding injection control and actual 64×5120 compressed-memory
text/image SSE checks. It establishes the tested tensor-input behavior, not
compressor interchangeability or completed production rollout. See the
[H20 handoff](h20-public-deployment.md) for the explicit configuration steps.

## Recovery and storage

`REMORY_MEMORY_DIR` contains snapshots on the inference host. The dedicated
launcher sets it to `RUNTIME_DIR/memory`. Each TP rank persists its own tensor;
the local session journal contains only snapshot references. Restore reloads
the remote snapshot and obtains handles for the new server epoch.

The previous durable snapshot is kept until the new local journal has been
atomically written. It is then deleted unless another retained artifact needs
it. Exported child checkpoints and opted-in training captures deliberately
preserve their referenced snapshots. Do not delete the inference snapshot
directory while retaining client journals that depend on it. Changing the
backend route, TP topology, checkpoint identity or snapshot mount is not a
transparent migration.

An old local tensor checkpoint is imported once on resume, then rewritten with
remote references. This compatibility import is the only tensor upload in the
new client. Legacy child checkpoints are checksum-checked before import.
Training captures use remote references; their tensor reader must run beside
the engine with its snapshot directory mounted. This change does not start or
deploy a remote trainer.

The default registry allowance is 2 GiB per TP rank, 256 live entries and a
one-hour lease, renewed by active clients. `REMORY_MEMORY_BYTES` changes the
byte allowance when starting the server. These bound resource use and do not
impose a 4096-token compression threshold.

## Validation

Run `python tools/run_memory_tests.py` with this workspace's existing controller
environment. The focused suite covers tensor/reference numerical equivalence,
more than 4096 slots, main replacement, failure rollback, child folding,
checkpoint recovery, cleanup retries, queue affinity and cancellation.

`tools/gpu_memory_smoke.py` is a local validation driver that must be launched
through the research resource queue. The successful TP1 run is recorded at
`validation/gpu-smoke-3/result.json`: real Stage-II-14925 encode, recursive
encode, snapshot save/load, composition, generation and release completed on
physical GPU 4. This is a protocol smoke test, not a model-quality benchmark.
Its owned server was stopped afterwards.

Earlier TP1 startup failures used incompatible CUDA assembler/compiler caches;
their receipts remain in `validation/gpu-smoke-job*`. TP2 startup and a minimal
NCCL probe failed before Remory execution because the container's 64 MiB
`/dev/shm` could not allocate NCCL proxy storage. The exact error and cleanup
record are retained under `validation/`. NVIDIA documents the container shared
memory requirement in its [NCCL troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html#shared-memory).
Real TP2/TP4 and remote H20 deployment are therefore not verified by these runs.
