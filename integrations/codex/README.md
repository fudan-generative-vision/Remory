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

## Agentic compaction (per session)

Enter `/agentic_compact` to enable message folding for the current session.
`/agentic_compact status` reports the setting; `/agentic_compact off` restores
standard Remory compaction. The setting persists when this session is resumed.
Enabling it does not compact immediately or change the automatic threshold.
Subsequent `/compact` and automatic compactions use the same folding path.

At that boundary, a temporary local Codex director uses native `spawn_agent`
with `fork_context=false`. Its organizer receives access to a deep copy of the
messages through four restricted tools: inspect, test, commit, and abort.
It uses the session's Remory model and reasoning setting. The director only
coordinates native tools; it does not make model calls or use a logged-in
Codex account. This organizer is a child of the temporary director, not an
extra research worker in the interactive thread.

The organizer proposes ordered replacements of assistant spans or entire tool
exchanges, with a factual summary for every removed span. User/system messages
are immutable; partial tool exchanges and untested candidates are rejected.
The parent revalidates the edit plan, encodes the **folded copy** as REMORY
residual memory, and durably saves the checkpoint before replacing its context.
Existing parent memory and completed regular-child memories are included in
the encode; the organizer's scratch conversation is not. Native `close_agent`
then closes the organizer, and its temporary home and transcript are removed.
The organizer creates no persistent residual checkpoint of its own.

Failures before commit retain the parent history. A failed close terminates the
isolated runtime and records a cleanup warning without invalidating a committed
parent checkpoint. Failed deletion of an uncommitted allocation records its
handle in the session provider directory for cleanup. Saved folding receipts
include the model, reasoning setting, source digest, edits and native thread IDs.
Structural checks cannot prove the semantic accuracy of an AI-generated summary.

The pinned upstream TUI does not register additional bare slash commands.
r-codex translates `/agentic_compact` at a fresh input boundary to a
`[Remory session] /agentic_compact` control message. Only an exact command in
the interactive thread's latest user input changes the setting; quoted history,
tool results and organizer messages cannot do so. The provider acknowledges it
without a model call. Start or resume through the updated r-codex
launcher to load the command; already-running terminal processes are unchanged.

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
