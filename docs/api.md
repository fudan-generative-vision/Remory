# API

SGLang listens on `http://127.0.0.1:8421` by default. Normal generation uses its
native `/generate` endpoint and response format. Remory adds saved memory to
that endpoint and a `/v1/compact` route on the same server.

Send a session header when creating, restoring, or deleting memory:

```http
X-Remory-Session: <session-id>
```

The session ID groups stored memories across requests. `GET /health` checks the
service, `/get_server_info` shows the SGLang configuration, and `/openapi.json`
describes the routes.
`GET /v1/remory/info` returns the actor and checkpoint identity, tokenizer and
summary configuration, and the context capacity used by `r-codex`.

## Generate

`POST /generate`

Before compaction, send an ordinary SGLang request:

```json
{
  "input_ids": [1, 2, 3],
  "sampling_params": {"temperature": 0, "max_new_tokens": 1024}
}
```

Native `text` prompts and streaming also work. For chat and tool use, render
messages with the actor's native tokenizer and chat template.

After compaction, add the checkpoint handle and send only the continuation IDs:

```json
{
  "input_ids": [8, 9],
  "sampling_params": {"temperature": 0, "max_new_tokens": 1024},
  "remory": {"handle": "rm_…"}
}
```

Remory restores the saved prefix, summary, and soft memory before SGLang
schedules the request. The model hook inserts the residual embeddings during
prefill. Requests with a handle accept one text sequence as `input_ids`;
batching, images, and SGLang KV sessions cannot be combined with the handle.

The response is SGLang's native `text`, `output_ids`, and `meta_info`, including
token usage and `finish_reason`. Set `stream: true` for native SSE output.
The Python and JavaScript clients expose non-streaming convenience methods.

## Compact

`POST /v1/compact`

```json
{
  "prefix_ids": [1, 2],
  "history_ids": [3, 4, 5],
  "summary_ids": [6, 7],
  "previous": null
}
```

Use the actor's native tokenizer and chat template for all token fields.
`prefix_ids` contains the system prompt, tools, and original task;
`history_ids` contains the messages being removed. Keep complete tool call/result
pairs together. Retained messages go into the next generation's continuation.

`Harness.compact` generates the summary through `/generate` before calling this
route. A harness with its own summary method can supply `summary_ids` directly.
Use the summary contract supplied by the checkpoint. The released Qwen encoder
uses a JSON summary matching `summary_contract.output_schema`; the provider
applies it during generation. The GLM example extracts a completed plain-text
answer before encoding memory.

Response:

```json
{"handle": "rm_…", "summary_ids": [6, 7], "receipt": {"…": "…"}}
```

Save the handle with the summary before replacing the old history. `receipt`
records token counts, source hashes, memory size, and compression depths.

For subsequent compactions, set `previous` to the old handle, keep the prefix
identical, and send only newly removed history. Qwen re-encodes the previous
summary and memory with that history, conditioned on the new summary. GLM keeps
the previous memory frontier, encodes the new history, and merges nodes when
needed. Each compaction creates a new immutable handle, so branches can keep
different checkpoints.

Qwen also accepts `branches: ["rm_…"]` to fold completed sub-agent memories into
the new checkpoint. Branches must belong to the same session and model. Their
summaries and soft tokens are re-encoded with the parent's history; the original
handles remain valid. The Codex provider tracks completed children and folds
each checkpoint once.

Internally, Qwen encoding posts to the same SGLang `/generate`
endpoint with `max_new_tokens: 0`, a Remory operation in
`sampling_params.custom_params`, and `return_hidden_states: true`. The hook
captures the selected actor layers and runs the memory network in the worker.
The hidden-state response channel carries the resulting memory rows.
GLM uses chunked prefill with a one-token generation budget and transfers encoded
rows through the server's binary cache, verified by checksums and completion
receipts. Both models expose the same compact and memory-handle API to clients.

## Stored memory

`DELETE /v1/memories/{handle}` deletes a checkpoint in the current session.
Handles have no automatic expiry. SQLite retains them across service restarts;
keep the database when resuming sessions. Reuse a handle with the same weights,
tokenizer, system prompt, and tool definitions.

A failed compaction leaves existing handles intact. Keep the original history
until a request succeeds. A cancelled request or retry can leave an unused handle
if the server has already finished; delete it after committing the chosen result.

| Status | Meaning |
| --- | --- |
| 400 | Invalid input, incompatible checkpoint, or context overflow |
| 404 | Memory handle not found in this session |
| 422 | Invalid request schema |
| 502 | Inference backend failed |

The Transformers reference server accepts the token-ID generation and memory
requests above; native text prompts and SSE are provided by the SGLang deployment.
