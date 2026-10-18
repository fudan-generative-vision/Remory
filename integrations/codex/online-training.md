# Online training in r-codex

Train REMORY from ongoing **Qwen3.8-27B** sessions, then select a memory checkpoint
without leaving the Codex terminal. Qwen stays frozen; only REMORY is updated.

Enter these commands directly in the **r-codex input box**. The leading character
is a backslash (`\`), not the slash used by Codex's `/compact` command.

| Command | Action |
| --- | --- |
| `\remory train` | Start or resume the shared background trainer. Wait for accepted samples and available training GPUs. |
| `\remory load` | Open the checkpoint picker. Use arrow keys and Enter to select; Cancel keeps the current version. |
| `\remory status` | Show the training phase, step, sample count, and this session's active/pending checkpoint. |
| `\remory stop` | Stop the shared trainer after its current step and save progress. Sample collection continues. |

These are local commands, intercepted before model input. They also work while
the agent is busy. Selecting a checkpoint takes effect at the **next successful
compaction** in this session. Existing memory tensors remain unchanged, and
other sessions retain their encoder versions. The picker includes the released
checkpoint so you can switch back.

After selection, the picker closes and reports checkpoint verification in the
transcript. `\remory status` and `\remory help` remain available during that work.
Other load/train/stop commands run in submission order; repeating a pending
command reports that it is already in progress instead of submitting it twice.
Choose Cancel or press Esc while the picker is open to keep the current version.

## One-time setup

Update your clone and [serving installation](../../docs/deployment.md#upgrade).
This feature needs the updated launcher **and a restarted server**. Use a shared
local directory for samples, checkpoints and training state; the server, client
and training worker must be able to access the same paths. Start with a new
directory when changing the base actor or released memory checkpoint.
If resuming existing sessions after installing a new runtime, also pass `--store`
with the existing server's `memory.sqlite` path; keep the same actor and base
checkpoint. The online training directory is separate from this memory database.

```bash
git pull --ff-only
python deploy/install.py --runtime-dir .remory-v0.4
```

Create the separate training environment described in the
[Qwen training guide](../../training#environment-and-models):

```bash
python3.12 -m venv .venv-training
# Install a compatible CUDA build of torch 2.9.1 first.
.venv-training/bin/python -m pip install -r training/requirements.txt
.venv-training/bin/python -m pip install --no-deps -e .
```

On GPUs that require a newer CUDA compiler than the training environment's
bundled Triton provides, set `TRITON_PTXAS_PATH=/path/to/cuda/bin/ptxas` before
launching r-codex. The worker inherits this setting. The compiler must support
your GPU architecture; a compatible PyTorch CUDA wheel alone is not sufficient.

The example reserves physical GPUs **0,1 for inference** and **2,3 for training**.
Change these indices for your machine. Each training GPU holds a full frozen
27B reader and trainable REMORY replica: these are **additional GPUs**, separate
from the inference allocation. The worker defaults to waiting for at least
100,000 MiB free and at most 1,024 MiB used on each training GPU. Longer contexts
can need more memory; these thresholds are availability checks, not a memory
guarantee.

Start SGLang from the repository root:

```bash
source .remory-v0.4/venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python -m remory.launch_server --tp 2 \
  --online-training-dir "$HOME/.local/share/remory/online-training"
```

In another terminal, from the same repository root:

```bash
source .remory-v0.4/venv/bin/activate
r-codex --online-training --training-gpus 2,3 \
  --train-python "$PWD/.venv-training/bin/python" \
  -C /path/to/project
```

GPU indices refer to `nvidia-smi` physical indices, independently of the client's
`CUDA_VISIBLE_DEVICES`. Full GPU UUIDs are also accepted. The launcher records
UUIDs and rejects overlap with the server's inference GPUs. For another session
using this configured pool, `r-codex --online-training -C /path/to/project` is
enough. `r-codex resume` remembers whether online training was enabled.
To enable it for an existing saved session, use
`r-codex resume SESSION --online-training` after configuring the server/pool.

Then, inside the r-codex input box:

```text
\remory train
\remory status
\remory load
```

Training does not start just by launching the CLI. `\remory train` starts it;
the worker can continue after the terminal closes. Run `\remory stop` from any
enabled session to stop that pool's trainer. It finishes an in-flight teacher
request or training step before stopping, so stopping is not immediate.

## What becomes training data

Only opted-in root sessions contribute. A sample contains the exact text-token
frontier before compaction, any earlier residual tensors, and the summary that
was actually installed. The client acknowledges a sample only after it saves
the new memory descriptor. Failed compactions, child-folding compactions, image
contexts, and manual compaction after a completed assistant answer are excluded.
Automatic compaction before the next assistant step supplies the normal samples.

The trainer generates a hypothetical continuation from that original frontier
using the serving model. It records the tokens; it **does not execute the
generated tool calls**. Incomplete continuations are excluded. It then minimizes
full-vocabulary reverse KL between the frozen reader with summary + residual
memory and the frozen reader with the original frontier. Teacher continuation
tokens are never part of the encoder's source history.

Sessions using the same pool share accepted samples. The default waits for two
samples, uses learning rate `1e-6`, and saves every 10 steps and at a clean stop.
Each sample is consumed once. A batch run handles up to 100 steps, then checks
for newly accepted samples. Weight, optimizer, sample-cursor and per-rank random
states are saved for resumption after interruption.

## Files and progress

All pool state lives in the directory passed to `--online-training-dir`:

| Path | Contents |
| --- | --- |
| `config.json` | Training interpreter, GPU UUIDs, learning rate, save interval and availability thresholds |
| `samples/` | Local tokenized history, summaries, residual tensors and teacher continuations |
| `worker.log`, `status.json` | Detailed errors and current progress |
| `checkpoints/*/serving/` | Loadable `model.safetensors`, configuration, checksums and training receipt |
| `checkpoints/*/training.pt` | Optimizer, consumed-sample IDs, step and RNG state |
| `tensorboard/` | Loss, gradient norm and scored-token count |

Stop the worker before editing `config.json`. Increasing `min_free_mib` makes it
wait for more available memory. `waiting_for_samples` and `waiting_for_gpu` are
normal idle states. A failed worker can be restarted with `\remory train`; check
`worker.log` for the underlying error first.

Captured history and checkpoints stay on your filesystem. They can contain
project content from the opted-in sessions. This release does not upload them
or include real conversation data in the repository. It checks finite gradients,
finite weights, checksums and architecture compatibility; it does not establish
that a newly trained checkpoint improves task quality. Evaluate a checkpoint
before using it for important work. GLM online training is not included.
