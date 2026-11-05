# Deployment

`python -m remory.launch_server` starts SGLang's native HTTP server with the
Remory model hook and compaction routes on the same port.
Qwen3.8-27B and GLM-5.3-Flash share one installation and SGLang revision:
`575759d90af942eeff89f1c8c33a1fcdb2da4181`. The checkpoint selects the model hook.
The context window defaults to the model maximum: 256K for Qwen and 1M for GLM.
The Transformers reference backend supports Qwen.

## Requirements

- Linux x86_64, Python 3.11+ with `venv`, `git`, and a C++ compiler.
- Budget at least 2 × 80 GB GPUs for Qwen3.8-27B + Remory and 100 GB of disk.
- A driver supporting CUDA 13.0.

The installer chooses the CUDA build and installs compiler components locally.
The launcher downloads model weights and compiles kernels on first use.
Downloads use GitHub, PyPI, PyTorch, NVIDIA, and Hugging Face.

SGLang loads the LLM across the selected GPUs using tensor parallelism. Each
worker loads a copy of the Remory network and uses its local actor hidden states
to encode memory. Text requests are handled serially.

## Install

```bash
python deploy/install.py
source .remory/venv/bin/activate
```

The installer creates `.remory/venv` and installs the pinned SGLang source with
the Remory hooks. Run it once; starting the server does not install packages.
The launch module reads the installed CUDA and kernel-cache paths automatically.

To put the environment on another disk or choose a CUDA build:

```bash
python deploy/install.py --runtime-dir /path/to/remory-runtime --cuda cu130
source /path/to/remory-runtime/venv/bin/activate
```

`--cuda` accepts `auto` (the default) and `cu130`. If an update changes
the pinned runtime recipe, install it in a new runtime directory.

### Upgrade

For this release, update the clone and install the new CLI in a fresh runtime:

```bash
git pull --ff-only
python deploy/install.py --runtime-dir .remory-v0.4
source .remory-v0.4/venv/bin/activate
python -m remory.launch_server --tp 2
```

Activate the same environment in the client terminal before running `r-codex`.
Start a new session after changing the server's base checkpoint; keep the old runtime and
database if you need to return to sessions created with the previous checkpoint.
When the actor and base checkpoint are unchanged, pass `--store` pointing to
the existing `memory.sqlite` to keep its saved sessions available after the
runtime upgrade. The [online training guide](../integrations/codex/online-training.md)
describes enabling `/remory train` and `/remory load`.

## Launch

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m remory.launch_server \
    --model-path Qwen/Qwen3.8-27B \
    --remory-checkpoint mocoV3/Qwen3.8-27B-REMORY-1.9B \
    --tp 2 \
    --host 127.0.0.1 --port 8421
```

`--model-path` accepts the checkpoint's target model ID or a local snapshot.
The published model ID resolves to the actor revision recorded in the checkpoint.
Both model arguments can be omitted to use the released Qwen defaults.
The memory checkpoint is pinned in [`models/load.py`](../src/remory/models/load.py).
Use the [upgrade steps](#upgrade) for an existing installation. Saved residual
memory belongs to the checkpoint that created it.

### GLM-5.3-Flash

Use the same environment and launcher, with the GLM model and memory checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m remory.launch_server \
    --model-path zai-org/GLM-5.3-Flash \
    --remory-checkpoint mocoV3/GLM-5.3-Flash-REMORY-1.24B \
    --tp 2 --host 127.0.0.1 --port 8421
```

The local evaluation used one FP8 GLM replica across **2 × 275,040 MiB L20D GPUs**
with tensor and expert parallelism. The Qwen requirement of 2 × 80 GB does not
apply to GLM's 320B base model. Adjust `--tp` to your hardware; the minimum GPU
count on 80 GB cards has not been measured. Allow at least 400 GB of disk for GLM.

Run the same generate → compact → continue example:

```bash
python examples/quickstart.py \
    --model-path zai-org/GLM-5.3-Flash \
    --remory-checkpoint mocoV3/GLM-5.3-Flash-REMORY-1.24B
```

Generation and residual encoding both use the native SGLang `/generate` route;
`/v1/compact` returns the same memory handle used by the Qwen example. Use
`GlmChatTemplate` with `Harness` when adapting Codex. It keeps GLM's native chat
format and removes completed reasoning from the summary.

The 1.24B checkpoint captures the collapsed attention inputs of GLM blocks
0/11/22/33/44 and emits 64 soft tokens per 1,024 history tokens. Subsequent
compactions preserve the old memory frontier, encode newly removed history, and
merge adjacent nodes as needed to stay within 4,096 soft tokens. The summary is
passed to the actor alongside memory; this encoder does not read the summary.

The runtime preserves the local evaluation's chunked prefill and eager execution.
It uses one active request per replica. The local evaluation set a 128K context
window; the public launcher defaults to the model's 1M maximum. Actual capacity
depends on the memory available for the KV cache.

### Other launch options

```bash
# Choose two GPUs.
CUDA_VISIBLE_DEVICES=2,3 python -m remory.launch_server --tp 2

# Optionally use a smaller context window to save GPU memory.
python -m remory.launch_server --tp 2 --context-length 32768

# Load local snapshots of the released models.
python -m remory.launch_server \
    --model-path /path/to/Qwen3.8-27B \
    --remory-checkpoint /path/to/Qwen3.8-27B-REMORY-1.9B \
    --tp 2
```

| Option | Default | Description |
| --- | --- | --- |
| `--tp` / `--tp-size` | `1` | Number of GPUs for tensor parallelism; use `2` for the Quick Start |
| `--context-length` | Model maximum | Input and generated tokens combined |
| `--mem-fraction-static` | `0.8` | SGLang static GPU memory fraction |
| `--base-gpu-id` | `0` | GPU index within `CUDA_VISIBLE_DEVICES` |
| `--host` | `127.0.0.1` | API bind address |
| `--port` | `8421` | API port |
| `--store` | `<runtime-dir>/memory.sqlite` | Session database |
| `--runtime-dir` | Detected from the environment | Installed CUDA paths, kernel cache, and session data |
| `--online-training-dir` | Disabled | Qwen online sample pool and encoder checkpoint versions; shared with r-codex |

Requests that exceed the context limit return an error. Reduce `--context-length`
if the model runs out of GPU memory. Run `python -m remory.launch_server --help`
for the supported options. A single GPU with enough memory can use `--tp 1`.

The original `./deploy.sh` shortcut remains available and uses this same launcher.

## Session data

The SQLite database at `.remory/memory.sqlite` stores summaries and residual
tensors across restarts; keep it with your session data. Model weights use the
Hugging Face cache.

When using a custom runtime directory or local weights, run the example with
that environment and those model paths:

```bash
python examples/quickstart.py \
    --model-path /path/to/Qwen3.8-27B \
    --remory-checkpoint /path/to/Qwen3.8-27B-REMORY-1.9B
```

## Transformers

For direct model inspection, a Transformers backend is also available:

```bash
python -m pip install -e '.[server,transformers]'
remory serve --backend transformers --device cuda:0 --store remory.sqlite
```

Both Qwen backends use the same compaction engine and verify checkpoint checksums.
See [development](development.md) for the SGLang hooks and backend interface.
