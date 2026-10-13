# Codex

`r-codex` opens the native Codex terminal UI with a local Remory provider.
Codex handles files, shell commands, approvals, and tools. Generation, summary
creation, and residual encoding run on the same SGLang service.

## Start a session

Follow the [deployment instructions](../../docs/deployment.md) and start the server:

```bash
source .remory/venv/bin/activate
python -m remory.launch_server --tp 2
```

In another terminal, activate the environment and open Codex:

```bash
source .remory/venv/bin/activate
r-codex -C /path/to/project
```

The first run downloads the pinned Codex 0.149.0 binary and the actor tokenizer.
The client does not download model weights. Use `--url http://host:8421` for an
existing server, or `--model-path /path/to/actor` for a local tokenizer.
No OpenAI account or user-provided API key is required.

On a separate client machine, install `python -m pip install -e '.[codex]'`
from the clone and run `r-codex --url http://host:8421 -C /path/to/project`.

Use `/compact` in the UI to create a summary and residual memory. Automatic
compaction uses the context capacity reported by the server. A failed summary
or encode leaves the existing context intact.

## Resume

```bash
r-codex resume          # Choose a saved session
r-codex resume --last
r-codex resume SESSION
r-codex list
```

Each session has its own Codex home, provider state, and transcript under
`~/.local/share/remory/sessions`. Set `REMORY_HOME` to choose another location.
The SGLang memory database must also persist across server restarts. A session
refuses to resume against a different actor or memory checkpoint.

Native sub-agents start from self-contained tasks with separate conversation
state (`fork_context=false`). With Qwen, a completed
child's summary and residual memory join the parent's next compaction; each
completed checkpoint is folded once. The default allows three direct children.

## Tools and options

Shell execution and file editing retain Codex's normal approval and sandbox
controls. `--unsafe` explicitly disables them. `--session NAME` names a new
session; `--codex /path/to/codex` uses an existing 0.149.0 binary.

To use search or document tools, supply a TOML file with your native Codex
`[mcp_servers]` entries:

```bash
r-codex --mcp-config /path/to/tools.toml
```

MCP namespaces and function arguments are preserved. External tools retain
their own installation and credential requirements. They are not bundled with
Remory. Image inputs are inspected by the deployed Qwen model in a separate
SGLang generation; the resulting textual observation is retained and compacted
with the conversation. Raw image tokens are not written into residual memory.

## Provider interface

The local provider implements `/v1/responses`, native `compaction_trigger`
requests, and `/v1/responses/compact`. It sends model requests to SGLang
`/generate` and memory requests to `/v1/compact`. Successful compactions return
an opaque `remory.v1.rm_…` item, whose descriptor is persisted before Codex can
replace its history. This is a local memory reference, not OpenAI ciphertext.

[`bridge.mjs`](bridge.mjs) remains a small example for callers implementing
their own Responses provider. The complete implementation is in
[`src/remory/codex`](../../src/remory/codex).

References: [Codex custom providers](https://developers.openai.com/codex/config-advanced/)
and [upstream Codex](https://github.com/openai/codex).
