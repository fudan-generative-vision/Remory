# Training Qwen3.8-27B + REMORY-1.9B

This directory contains the Stage I and Stage II trainers used for the released
[Qwen memory checkpoint](https://huggingface.co/mocoV3/Qwen3.8-27B-REMORY-1.9B),
their local dependencies, portable configuration helpers, and
[readable data examples](examples). **Only Qwen3.8-27B training is covered.**
Both stages freeze Qwen and train only the memory network; there is no target LoRA.

For training from live r-codex sessions with `\remory train` and switching memory
versions with `\remory load`, see [online training](../integrations/codex/online-training.md).

| | Stage I: reconstruction | Stage II: residual memory |
| --- | --- | --- |
| Input | One contiguous 8K–64K token document window | Causal history, history-only handoff summary, and assistant continuation |
| Teacher | Raw document → repeat the document | Raw history → continuation |
| Student | Soft memory → repeat the document | Summary + soft memory → the same continuation |
| Initialization | Pinned pretrained DFlash2 backbone | Stage I checkpoint; new summary-attention and depth modules |
| Loss | Full-vocabulary reverse KL, student ∥ teacher, T=1 | Same loss, with response-position weighting |
| Peak / minimum LR | 1e-5 / 1e-6 | 1e-4 / 1e-5 |
| Update | 8 documents; scored-token mean | Whole responses totaling ≥16,384 target tokens and ≥4 responses; short tail merged |
| Schedule | 500-step warmup, cosine decay, one epoch | 500-step warmup, cosine decay, five planned epochs |

Both stages use AdamW (betas 0.9/0.95, weight decay 0.01) and seed 20260917.
Stage II forms fixed batches from a context-length order, then shuffles **batch
order**, preserving membership, independently each epoch.

The release was initialized from Stage I **step 8000** and exported at Stage II
**step 14925**. Five epochs describes the configured schedule, not a claim that
the export completed all five. The original Stage I corpus selected approximately
3B source tokens. The helpers below compute the schedule from your own corpus;
the example inputs do not reproduce the original dataset.

## Environment and models

Run commands from the repository root. Use a separate Python 3.12 environment
from the inference/SGLang environment:

```bash
python3.12 -m venv .venv-training
source .venv-training/bin/activate
# Install torch 2.9.1 with a CUDA build compatible with your system first.
# The source environment used torch 2.9.1+cu130.
python -m pip install -r training/requirements.txt

export TARGET_MODEL=/models/Qwen3.8-27B
export DFLASH2_MODEL=/models/Qwen3.8-27B-DFlash2
hf download Qwen/Qwen3.8-27B --local-dir "$TARGET_MODEL"
hf download incoai/Qwen3.8-27B-DFlash2 \
  --revision 015e795645c74b1a0eeef3b570031fb62e769bc5 \
  --local-dir "$DFLASH2_MODEL"
```

The helpers verify the pinned DFlash2 config/weight checksums and record
source/tokenizer/data hashes. Keep the same local target and tokenizer for both
stages. Model licenses apply separately. Data preparation needs only tokenizer
files; training needs the complete target weights.

The launchers use **four CUDA GPUs**. The reference `dp` mode replicates the
frozen 27B model and trainable memory network on each GPU. The source `tp` mode
shards the reader across four GPUs. Memory depends on context lengths and execution
settings; the two-GPU inference recipe is not a training-memory specification.
This publication was checked on CPU, without rerunning full GPU training.

Use `TRAIN_PYTHON=/path/to/python bash training/run.sh ...` for an existing
interpreter. `stage1 --help`, `stage2 --help`, and `prepare --help` do not launch
workers. Output directories/config files must be new; failed preparation can
leave partial files, so retry with a fresh output directory.

## Inspect the data format on CPU

The JSONL files are short, synthetic format examples, with no source-corpus or
benchmark records. This explicitly enabled demonstration writes the same SQLite
and manifest formats read by the trainers:

```bash
bash training/run.sh prepare stage1-data \
  --input training/examples/stage1.documents.jsonl \
  --target-model "$TARGET_MODEL" \
  --output-dir training/prepared/example-stage1 --allow-short-examples

bash training/run.sh prepare stage2-data \
  --input training/examples/stage2.continuations.jsonl \
  --target-model "$TARGET_MODEL" \
  --output-dir training/prepared/example-stage2 --allow-short-examples
```

These fixtures are rejected by production configuration helpers. See
[examples/README.md](examples/README.md) for fields and token boundaries.

## Stage I

Prepare document JSONL with `id`, `text`, and optional `split` (`train` or
`validation`). Without an explicit split, a document hash assigns about 1% to
validation. Provide both splits. Duplicate documents, including overlap between
splits, are rejected. Documents need ≥8192 target-tokenizer tokens; a deterministic
contiguous window of at most 65536 tokens is retained.

```bash
bash training/run.sh prepare stage1-data \
  --input /data/documents.jsonl --target-model "$TARGET_MODEL" \
  --output-dir training/prepared/stage1

bash training/run.sh prepare stage1-config \
  --manifest training/prepared/stage1/repeat.manifest.json \
  --target-model "$TARGET_MODEL" --pretrained "$DFLASH2_MODEL" \
  --output training/prepared/stage1/init.json

CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/run.sh stage1 \
  --pretrained-init training/prepared/stage1/init.json \
  --manifest training/prepared/stage1/repeat.manifest.json \
  --output-dir training/runs/stage1 --mode dp --projection-tokens 1024 \
  --reader-checkpoint-after-tokens 0 \
  --save-interval 100 --validation-interval 500 --keep-last-checkpoints 3
```

For a small corpus, explicitly set `--warmup-steps` on the config command below
the calculated total updates. No schedule is silently shortened.

The source FinePDFs-Edu parquet importer is also available through
`bash training/run.sh repeat-data --help`. It applies an education-score threshold
and preserves a source receipt. The JSONL path above does **not** apply that filter,
assemble the original 3B-token cohort, or download the source corpus.

## Stage II

Prepare rows containing `id`, `messages`, and `summary`. The **last assistant
message** is the held-out continuation; all prior messages form the causal
history. The supplied summary must use only that history. The importer preserves
Qwen's native system/tool prefix and thinking boundaries. Production inputs need
≥8192 compressible history tokens per row and at least four events.

```bash
bash training/run.sh prepare stage2-data \
  --input /data/continuations-with-handoffs.jsonl \
  --target-model "$TARGET_MODEL" --output-dir training/prepared/stage2

export STAGE1_CHECKPOINT=training/runs/stage1/checkpoint-00008000
bash training/run.sh prepare stage2-config \
  --manifest training/prepared/stage2/continuations.manifest.json \
  --summary-manifest training/prepared/stage2/summary.manifest.json \
  --selection training/prepared/stage2/selection.json \
  --context-order-plan training/prepared/stage2/context-order.json \
  --target-model "$TARGET_MODEL" --pretrained "$DFLASH2_MODEL" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --output training/prepared/stage2/init.json

CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/run.sh stage2 \
  --pretrained-init training/prepared/stage2/init.json \
  --context-order-plan training/prepared/stage2/context-order.json \
  --output-dir training/runs/stage2 --mode dp --feature-residency gpu \
  --node-batch-size 64 --projection-tokens 1024 \
  --reader-checkpoint-after-tokens 0 --reader-layer-token-budget 560000 \
  --reader-retain-min-tokens 24576 --short-response-node-batch-size 32 \
  --fused-adam --save-interval 25 --keep-last-checkpoints 3
```

Select an existing Stage I checkpoint; 8000 is the release's parent step, not a
requirement for custom data. Stage II initializes a **fresh optimizer**. This
differs from resuming Stage II with its optimizer, RNG states, and data cursor.

Imported handoffs are marked **caller-provided**, not model-generated. Checksums
bind them to their causal histories, but cannot prove that supplied text contains
no information from the future answer. Generate or write summaries before exposing
the held-out answer to the summarizer.

### Original trajectory and summary pipeline

The original Stage II used reasoning/coding trajectories, length-qualified events,
and offline structured summaries from frozen Qwen servers. These utilities are
also included; run `bash training/run.sh COMMAND --help` for their options:

| Command | Purpose |
| --- | --- |
| `allturn-data` | Render accepted assistant turns from the source xhigh and coding schemas |
| `long-history` | Keep events with ≥8192 history tokens; derive nonoverlapping 8192-token continuation windows inside eligible long reasoning, with a 256-token known recent tail |
| `summaries` | Generate causal Codex-style JSON handoffs through an explicit SGLang `/generate` endpoint |
| `select-summaries` | Freeze an audited selection when events lack valid summaries; record exclusions |
| `curriculum` | Build the immutable context-length order used to form batches |

For example, given a derived long-history manifest:

```bash
bash training/run.sh summaries \
  --manifest /data/long-history.manifest.json --target-model "$TARGET_MODEL" \
  --cache /data/summaries.sqlite --output-manifest /data/summary.manifest.json \
  --base-url http://127.0.0.1:8421 --workers 4 --summary-format json
```

This uses an existing server with the same frozen target and exact output token
IDs. Truncated/invalid summaries are rejected, completed rows survive interruption,
and incomplete caches are never silently treated as complete. `--check-only`
validates inputs without contacting a server. Partial selection is a separate
explicit operation; the custom JSONL path creates a complete selection automatically.

Training uses an **offline full-causal-history rebuild** for each event, with
one summary conditioning all memory levels. It compresses 1024-token blocks into
64 slots, keeps a 4096-slot frontier, and merges the oldest adjacent equal-depth
nodes when needed. It is not an online rollout trainer backpropagating through
successive agent compactions.

## Checkpoints, resume, and checks

Runs write their configuration, JSONL/TensorBoard metrics, and
`checkpoint-XXXXXXXX/` directories. Checkpoints contain memory parameters,
optimizer state, per-rank RNG states, and the exact data cursor. Frozen target
weights are not copied into each checkpoint. These are training checkpoints,
not the packaged Hugging Face inference export.

To resume either stage, keep the existing output directory and the same data and
execution arguments. Replace `--pretrained-init ...` with, for example:

```bash
--resume-from training/runs/stage2/checkpoint-00000100
```

Resume checks run contracts, data hashes, and batch boundaries. Keep recorded
paths/files unchanged; regenerating an initialization config creates a new run.
Stage II has an explicit `--allow-execution-change` migration option; read its
help before changing execution settings for an existing run.

For a bounded GPU smoke run, append `--max-additional-steps 1` to a training
command. It still loads the full models on four GPUs. CPU tests load no model
weights and contact no server:

```bash
bash training/run.sh test -q
```

Tests cover reverse-KL values/gradients, causal grouped convolution, gradient
checkpointing, shuffled batch membership and resume boundaries, summary-cache
integrity and failure recovery, and data import/held-out-answer separation.

On 2026-10-15, all **51 CPU tests** passed in the pinned training environment.
Both JSONL examples also passed preparation and cache checks with the real Qwen
tokenizer. Generated production configs matched the source runs' architecture,
objective, optimizer, and schedule; both full-size architectures were checked
without allocating model weights, using PyTorch's meta device. This validation
does not replace a distributed GPU training run.

## Source provenance

Dependency snapshots are byte-preserved and listed with SHA-256 in
[source-manifest.json](source-manifest.json). Stage I comes from the 3B-token
reconstruction run; Stage II comes from the lr1e-4 run with per-epoch fixed-batch
shuffling. `prepare.py`, `run.sh`, portable config templates, and synthetic data
are additions for this release.

Some transitive helpers retain historical `qwen35`, `qwen36`, or `glm52` filenames;
the Qwen entry points import shared model/data/checkpoint helpers from them.
Their presence does not add a GLM training recipe. The serving package stays
separate. See [third-party notices](../NOTICES.md).
