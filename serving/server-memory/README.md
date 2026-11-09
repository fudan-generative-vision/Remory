# REMORY serving runtime

This directory contains the advanced SGLang runtime and request gateway used by
REMORY's local and remote inference pool.

- [Mixed pool, encoder optimizations and measurements](docs/inference-pool.md)
- [Server-owned memory and durable checkpoints](docs/server-memory.md)
- [H20 installation and deployment](docs/h20-public-deployment.md)

For the standard installation and Codex quick start, use the
[root README](../../README.md). The command-line recipes below describe the
historical evaluation stack; machine-specific research sessions are not included.

<details>
<summary>Historical evaluation runtime reference</summary>

# Recursive Memory

SGLang + residual Compressor + **Codex native compact / xhigh**, with two supported uses:

1. Install the environment and evaluate **Terminal-Bench 2.1 / AutomationBench / JobBench / BrowseComp100**.
2. Open the **Codex terminal interface with the same model, compression and tools as TB**.

The inference dependencies, source checksums, licenses and regression tests
needed by these two workflows are retained. The separately published
[training source snapshot](training/README.md) contains the current repeat →
summary-conditioned residual training implementation and its CPU tests. It is
not installed by `install.sh` and does not change the evaluation harness.
Old demos, unsupported benchmark runners and duplicate legacy evaluation launchers
are not distributed.

The [2026-09-18 progress record](docs/progress-20260918.md) covers the long-document
Compressor-only training run, standalone summary generation, the paused Stage 2
checkpoint and recovery of the ongoing native-only TB2.1 evaluation.

There is one harness profile: Codex controls compaction timing; its local
`/compact` implementation produces a text summary plus recursive residual soft
memory. Qwen's native chat template is explicitly pinned to `xhigh`. WM-triggered,
summary-only and low/medium sweep launch profiles are not distributed.

## Install

Linux x86_64 and Python 3.12 + venv are required. CUDA JIT also needs Python's
development headers (`python3.12-dev`, installed by `--system-packages`). Installation probes the CUDA
driver directly: Ampere/Hopper use the pinned CUDA 12.8 build; SM 10.3 uses CUDA
13.0 and requires a compatible driver (580.105.08 was validated). A private
nvcc/ptxas toolchain and cuRAND headers are installed inside the prefix.
GPU allocation is detected at launch: set `CUDA_VISIBLE_DEVICES=0,1` to select
cards, or omit it to use all GPUs visible inside the worker/container. The
singular `CUDA_VISIBLE_DEVICE` is accepted as an alias. Supported TP counts are
1/2/4/8, subject to sufficient VRAM; unsupported counts fail rather than silently
dropping cards. The reference setup is 2 × H800 80 GB; other GPU allocations,
including 8 × RTX 4090 24 GB, still need a target-machine GPU smoke test.
Reserve ample disk space for weights, environments and task images.

```bash
git clone https://github.com/1ring2rta/recursive-memory.git
cd recursive-memory
# The Compressor is public; HF_TOKEN and Codex login are not required.
bash install.sh --config deployment.example.json --prefix /data/rm-stack --system-packages --smoke
```

The installer downloads the pinned Qwen3.8-27B and
[residual Compressor step2320](https://huggingface.co/mocoV3/qwen38-residual-compressor-step2320),
verifies hashes, installs isolated environments, and runs a synthetic serving
smoke test covering stochastic sampling, recursive memory and a local document
conversion. No private Git submodule is required. `--system-packages` authorizes
apt installation; omit it if host dependencies are already present.

For local weights or different GPU IDs/port, copy and edit
[`deployment.example.json`](deployment.example.json), or pass
`--target-path /absolute/path/to/Qwen3.8-27B`. The example downloads weights
directly from `hf-mirror.com`, removing proxy variables only in that download
subprocess; other installation traffic retains your configured proxy. The
installer never starts or changes a VPN. Credentials belong in the environment
or a literal `--env-file` for runtime commands. `--dry-run` previews without downloads or writes;
omitting `--smoke` avoids starting a GPU server during installation.

The `server` section accepts `devices`, `tp`, `memory_fraction`, `context_length`,
`max_total_tokens` and `cuda_graph` (all default `auto`), plus
`max_running_requests` (default 1). Auto profiles use detected VRAM to estimate
model headroom, with a 262144 context/token pool on large multi-GPU allocations
or a single GPU with at least 128 GiB, and 65536 otherwise. These estimates are not an OOM guarantee; explicit settings
remain available. Concurrent residual
serving requires the updated SGLang process: encode forwards remain singletons,
while request-isolated recovery/discrete generations may batch. The client keeps
legacy servers exclusive and accepts concurrency only with the new serving ABI.
Changing these values requires a server restart; an existing server is never
silently replaced. The Compressor is replicated per rank: total card memory
is not a single shared VRAM pool. GPU masks apply when starting a new server;
reusing an existing server does not move it to different cards.

For large L20D allocations, use **one TP1 replica per GPU, two requests per
replica and CUDA Graph**. The complete Chinese walkthrough is
[docs/reproduce-l20d.md](docs/reproduce-l20d.md). This is the configuration tested
on this deployment; plain `run` continues to use one tensor-parallel server.

```bash
# Only the installation smoke uses GPU 0; the following node launch uses all visible GPUs.
CUDA_VISIBLE_DEVICES=0 bash install.sh --config deployment.l20d.example.json \
  --target-path /models/Qwen3.8-27B --prefix /data/rm-stack --system-packages --smoke
bash stack.sh node --prefix /data/rm-stack --env-file /secure/rm.env \
  start --node-id node-a --gpus auto --concurrency 2 --cuda-graph
```

Four visible GPUs create eight task workers. `node plan` and `node check` inspect
the allocation without starting a server or claiming a task. `node start`
requires a successful installation smoke and validates the document sandbox,
actual CUDA Graph mode, request capacity and compact threshold before work.
Native compact remains at 179200 with the 262144 context / 524288 token pool.
The same queue can resume with a new serving deployment; endpoint-only changes
are archived when an interrupted task moves to a different replica.

To keep models running independently of benchmarks, use `node serve` and the
[shared request gateway](docs/shared-serving.md). Other projects can share its
OpenAI-compatible or native SGLang endpoint; requests queue by project while
each benchmark keeps its own tasks and results.
All four `stack.sh run` benchmarks accept `--memory residual` (default) or
`--memory none` for the paired native text-summary ablation. Use distinct pool
names for the two modes; the shared model service can serve both.

## Evaluate

```bash
export SERPER_API_KEY='your-key'
export MINERU_API_KEY='your-key'

bash stack.sh run --prefix /data/rm-stack tb21 --start-server
# Or, on an idle worker allocation:
bash stack.sh run --prefix /data/rm-stack automationbench --start-server
# Document/workplace tasks, or the frozen random 100-question browsing subset:
bash stack.sh run --prefix /data/rm-stack jobbench --start-server
bash stack.sh run --prefix /data/rm-stack browsecomp100 --start-server
```

These commands initialize/resume their own shared-filesystem pools. Multiple
machines can use the same shared installation prefix, with one local server
per worker allocation. `--once` runs one task per slot; `--task TASK_ID` selects a separate subset
pool. Existing local experiment pools are never migrated or cleared.

To overlap tool execution across two samples while sharing one SGLang server:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash stack.sh run --prefix /data/rm-stack tb21 --start-server --concurrency 2
```

Omit the GPU variable to use every visible card. `--concurrency` defaults to 2;
it creates independent task slots in the same pool, not additional model copies.
An owned server is configured for at least that many requests; a reused server
must already support the concurrent residual ABI and capacity. Model VRAM,
CPU/RAM and task isolation still constrain useful concurrency.

Existing TB pools resume from their frozen local task references without a
registry lookup. Network failures pause affected slots before they claim more
tasks; APT bootstrap has bounded transport fallback, and transient SGLang
readiness/metadata requests retry without relaxing checkpoint validation.

| Profile | Tasks | Attempts | Budget |
|---|---:|---:|---|
| TB2.1 | 89 | 1/task | 24-hour agent phase; task-specific official verifier budget |
| AutomationBench public 1.0.6 | 600 | 1/task | 1 hour/task; no step limit |
| JobBench main | 65 | 1/task | 24 hours/task |
| BrowseComp100 | 100, seed 20260912 | 1/task | No task time limit |

JobBench and BrowseComp100 use the same native compact + residual + xhigh
generation profile and default concurrency 2. Their datasets download only when
their pools are prepared, never during package installation. Data and results are
not shipped in Git. See [JobBench / BrowseComp reproduction](docs/jobbench-browsecomp.md)
for the convenience launchers, private task artifacts, monitors and **separately
started judges**. JobBench uses the pinned official rubrics with a logged-in Codex
transport; BrowseComp uses local Qwen first, then Codex live-web appeals for all
rejections. These are distinct, documented judging protocols.

These are local protocols, **not a claim of official leaderboard equivalence**.
TB uses official task/verifier artifacts through the Docker-free PRoot adapter.
PRoot is neither identical to Docker nor a security boundary.

## Interactive Codex

```bash
bash chat.sh --prefix /data/rm-stack --workspace /path/to/your/project --start-server
```

Type successive user messages in the normal Codex terminal UI. An optional
`--prompt 'Explain this project'` supplies the first message. This uses TB's
checkpoint, model catalog/system instructions, Responses gateway, **native
compact + residual memory + xhigh**, vision, `update_plan`, Serper search,
Docling and MinerU. There is no fixed-round WM trigger; physical residual
compression blocks remain 1024 tokens. The default 179200-token compact trigger
is lowered when a worker's actual SGLang context capacity is smaller.

Interactive host commands use `workspace-write` and `on-request` approvals by
default. Unlike evaluation, they do not run inside a benchmark task rootfs.
`--unsafe` explicitly disables these protections. No personal Codex login or
subscription is needed; model requests go to local SGLang. Paid search/document
providers still use your keys. MinerU sends documents to its external service.

## Monitor

```bash
bash stack.sh monitor --prefix /data/rm-stack all --port 8507
# Optional separate benchmark views:
bash stack.sh monitor --prefix /data/rm-stack tb21 --port 8511
bash stack.sh monitor --prefix /data/rm-stack automationbench --port 8508
bash stack.sh monitor --prefix /data/rm-stack jobbench --port 8509
bash stack.sh monitor --prefix /data/rm-stack browsecomp100 --port 8510
```

Forward these loopback ports over SSH. Trajectories, tool responses, images,
memory archives and results remain under the installation prefix.

If official verifier dependencies fail to download, the agent's final filesystem
snapshot is retained as `verification_pending`, not a false score of zero.
Resume verification without rerunning the model:

```bash
bash stack.sh verify --prefix /data/rm-stack tb21 --job YOUR_POOL_NAME
```

This CPU-side queue only auto-restores compatible filesystem-only tasks.
VM/process state is not contained in a filesystem snapshot; such tasks require
an explicit operator decision. Agent timeouts with a valid official verifier
reward remain scored, including reward 0 in the mean.

All launchers reuse an attested server or stop only the server they started.
They never restart a different checkpoint on an occupied port. Do not run chat
and evaluation concurrently on the same inference device. Configure any proxy
externally; the installer does not change VPN connections.

See the [deployment guide](docs/evaluation-stack.md) for details and limitations.
Recent fixes and migration notes are in the [changelog](docs/changelog.md).

## Source checks

```bash
python -m pip install -e '.[test]'
pytest -q
```

The controller package has no heavyweight dependencies; the installer owns
isolated serving/controller/document environments. CPU tests do not substitute
for a clean GPU installation or a complete benchmark run.

The TB2.1 node launcher also supports a paired [native compact without residual memory](docs/native-compact-ablation.md) control with `--memory none`.

</details>
