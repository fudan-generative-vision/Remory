# Training data examples

Both files contain **synthetic, hand-written format examples**, not records from
the original training corpus or evaluation benchmarks. They are deliberately
short and require `--allow-short-examples` for CPU preparation. Do not train on
these two-row fixtures.

## Stage I: documents

[`stage1.documents.jsonl`](stage1.documents.jsonl) has one training document and
one disjoint validation document. Each line is an object:

```json
{
  "id": "synthetic-train-001",
  "split": "train",
  "text": "A stable sorting algorithm preserves the input order of elements with equal keys..."
}
```

`id` and `text` are required; `split` and `url` are optional. Production documents
need ≥8192 Qwen tokens. The importer tokenizes once, keeps one contiguous window
of at most 65536 tokens, rejects duplicate text, and writes `repeat.sqlite` plus
`repeat.manifest.json` with source/tokenizer/database hashes.

The teacher reads the raw window and repeat instruction. The student reads its
compressed soft memory and the same instruction. Both score the document tokens
and end-of-message token using the frozen target model.

## Stage II: history, summary, continuation

[`stage2.continuations.jsonl`](stage2.continuations.jsonl) has two coding tasks.
Each line has this shape (values abbreviated here):

```json
{
  "id": "synthetic-coding-001",
  "messages": [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Fix first_even(values)..."},
    {"role": "assistant", "reasoning_content": "Check edge cases...", "content": "Observed StopIteration..."},
    {"role": "user", "content": "Please show the fix and tests."},
    {"role": "assistant", "reasoning_content": "Use a default...", "content": "def first_even(values): ..."}
  ],
  "summary": {
    "current_progress": ["Inspected first_even and observed failing cases."],
    "key_decisions": ["Preserve the first-match order."],
    "important_context_constraints_preferences": ["Return None when there is no even integer."],
    "next_steps": ["Implement a default result and show tests."],
    "critical_data_examples_references": ["[3, 4, 2] currently returns 4."]
  }
}
```

Only the **final assistant message** is scored. Earlier assistant messages, user
messages, and tool results belong to history. `reasoning_content` is optional;
when present, Qwen's native thinking template preserves it. Optional `tools`,
assistant `tool_calls`, and tool messages with `tool_call_id` use the source
renderer's OpenAI-style schema. Messages must contain textual `content`.

All five summary fields are required arrays of strings with at most 12 items.
`current_progress` and `important_context_constraints_preferences` must be
nonempty. Summaries have a 4096-token cap and must describe only history,
excluding the final assistant's reasoning and answer. The importer records them
as caller-provided; it does not generate them.

The native system/tool prefix stays outside compression. The teacher sees the
raw causal history; the student sees the handoff and soft memory. Both receive
the same assistant opening and teacher-forced continuation prefix. Prompt and
response are tokenized separately at the native generation boundary; the held-out
answer never enters the summary request.

Preparation writes `continuations.sqlite`, a sealed `summaries.sqlite`, their
manifests, `selection.json`, and `context-order.json`. The JSON metadata makes
coverage, tokenizer identity, and checksums inspectable. See the
[training guide](../README.md#inspect-the-data-format-on-cpu) for commands.
