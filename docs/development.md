# Development

```bash
python -m pip install -e '.[test]'
pytest -q
node --test integrations/test.mjs
```

Install `.[test,transformers]` to include the model and memory-hook tests.
These run locally without downloading the released 27B model. An optional GitHub
Actions configuration is in [ci-tests.yml](ci-tests.yml).

## Inference code

[`engine.py`](../src/remory/engine.py) implements compaction and generation.
[`pyramid.py`](../src/remory/pyramid.py) keeps memory within the slot budget, and
[`store.py`](../src/remory/store.py) persists immutable checkpoints in SQLite.

[`launch_server.py`](../src/remory/launch_server.py) parses launch options and
configures the installed CUDA toolchain before importing SGLang.
[`sglang_server.py`](../src/remory/sglang_server.py) extends SGLang's native HTTP
server. Normal answers, summaries, and residual encoding use `/generate`.
The Qwen [hook](../src/remory/backends/sglang_hook.py) captures decoder outputs
with a complete prefill and returns packed memory rows. It disables shared
prefix caching and uses one active request.
The GLM [hook](../src/remory/backends/glm_hook.py) captures mHC attention inputs
across prefill chunks, scatters memory at absolute input positions, and confirms
consumption after the complete prefill. Its internal binary cache is separate
from the durable, session-bound SQLite store. Each request has a unique cache
salt. Both paths disable CUDA graphs and overlapping schedules.

SGLang is pinned to `575759d90af942eeff89f1c8c33a1fcdb2da4181` for both models.
The [patch](../deploy/sglang.patch) initializes the Qwen hook, returns packed
memory rows, and connects the GLM adapter and native image processor. The installer
and launcher verify all eight files against the [recipe](../src/remory/backends/sglang_recipe.json).

`Harness.compact` generates the summary, then calls `/v1/compact` to build and
save residual memory. That route uses the same server's native `/generate`
endpoint for each encoding pass. A subsequent `/generate` request with a
`remory.handle` restores the summary and embeddings before SGLang schedules it.

For another inference engine, implement `Backend.encode` and `Backend.generate`
in [`types.py`](../src/remory/types.py). The shared engine handles compaction,
recursion, and storage; the backend supplies encoding and embedding-aware generation.

## Tested setup

The 2026-10-18 r-codex follow-up fixes passed all 89 runtime tests. Native Codex
0.149.0 terminal checks with fixture inference covered Unicode input, window
resizing, the r-codex exit hint, and UUID resumption into the same saved session.
They also covered checkpoint verification, duplicate-command acknowledgement,
queued commands, and status queries during verification and model generation.
Picker cancellation and late responses are covered by the transport tests.

On 2026-10-18, the online-training changes passed 131 Python tests, including
the training suite. A native Codex 0.149.0 terminal smoke test exercised all four
`\remory` commands, the checkpoint picker, and command input during a busy turn.
That terminal test used fixture inference and verified that local commands never
reached the model.

A separate GPU smoke test used the full frozen Qwen3.8-27B reader and released
REMORY checkpoint for one optimization step on a short synthetic frontier.
It produced finite, nonzero memory gradients and changed 110 of the encoder's
117 tensors while keeping the reader frozen. Resuming the saved optimizer/RNG
state repaired deliberately interrupted publication pointers without repeating
the consumed sample. This validates the update and resume paths, not downstream
task quality or long-context training memory requirements.

The updated public SGLang server also passed a live short-input check on an
independent GPU: capture and acknowledge a compaction, generate a complete
teacher continuation, select the trained checkpoint, and compact and generate
with its weights. Existing memory tensors remained unchanged and usable;
returning to the released encoder reproduced the original residual tensors.
Another session retained its original selection. The temporary server was
stopped after the check.

On 2026-10-13, a fresh installation of the shared SGLang runtime passed 51 Python
tests and dependency checks, using Python 3.12, PyTorch 2.13.0, Transformers
5.12.1, and CUDA 13.0. Four JavaScript tests also passed with Node 22. The installer
verified all eight patched SGLang files and the CUDA component checksums.

The released Qwen encoder's 117 tensors match the source weights exactly. The
Hugging Face loader and this runtime produce bitwise-identical CPU outputs to
the source for short, padded, and recursive-depth inputs.

The new Qwen release also passed a live native Codex 0.149.0 session through the
public launcher: shell execution, `apply_patch`, `/compact`, exit and resume with
the same residual handle, and `view_image`. The model recalled the file contents
after compaction and identified the test image. This smoke test used one L20D
GPU with a 32K context setting and short inputs. It does not measure full-context
behavior or the 2 × 80 GB deployment budget. Sub-agent folding and deduplication
are covered by the engine and provider tests.

The released 1.24B GLM encoder produced bitwise-identical CUDA outputs to the
local evaluation encoder for 37-token and 1,024-token source blocks. Eight
reference tests also passed for mHC feature capture, chunked memory injection,
request alignment, and vision-prefix validation. The full 320B GLM actor has
not been rerun through the public launcher; the integration ports the previously
evaluated local runtime on this same SGLang revision. The GPU checks used an
existing compatible environment and test-only settings for the shared node's
NCCL and available memory.
