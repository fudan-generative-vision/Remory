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
Local chat, shell/file tools, goals, sub-agents, and Remory compaction do not
require an OpenAI account or an external API key. The research tool profile
adds Serper search and document parsing; its credentials are described below.

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
The launcher routes `/agentic_compact` through its local native RPC command
channel, shared with `/remory`. The command and acknowledgement never enter
the model's history. Only a standalone command from the interactive thread
changes the setting; quoted history, tool results and child messages cannot
do so. Mutating commands wait for an active turn to finish; status is read-only.
Start or resume through the updated r-codex launcher to load the command;
already-running terminal processes are unchanged.

### Local deployment with legacy sessions

`python -m remory.codex.local_entry` optionally reads `.remory/launcher.json`
(or `REMORY_LAUNCHER_CONFIG`). New sessions use this version and the configured
default server. `r-codex list` and the resume picker combine both session stores,
using `remory:SESSION` and `legacy:SESSION` to disambiguate names. Existing sessions
retain their saved runtime, model and memory backend; no implicit migration takes
place. `r-codex --legacy ...` explicitly selects the previous launcher.

The optional `remory.codex.legacy_bridge` provides the public memory API over an
attested legacy residual server. It runs on CPU, preserves the checkpoint's
compression algorithm, and stores only new immutable memory handles. It does
not load model weights or restart shared inference. Its SQLite memory store
must persist for new sessions to resume. Server identity changes fail closed.
This adapter does **not** advertise online training; `/remory` training commands
require a server that supports the v0.4 online API. Agentic compaction works
through this adapter. Legacy sessions continue to use their original features.

Deployment configuration contains paths and endpoints, not credentials. Optional
`tools_env_file` reads only literal Serper/MinerU assignments and never executes
shell code. `tools_venv` reuses the installed document tools. If `research_tools`
is enabled in this local configuration, `--no-research-tools` starts a new
session without them.

## Resume

```bash
r-codex resume          # Choose a saved session
r-codex resume --last
r-codex resume SESSION
r-codex resume NATIVE_THREAD_UUID
r-codex list
```

Each session has its own Codex home, provider state, and transcript under
`~/.local/share/remory/sessions`. Set `REMORY_HOME` to choose another location.
The SGLang memory database must also persist across server restarts. A session
refuses to resume against a different actor or memory checkpoint.

The picker supports Unicode search, arrow keys, and Enter. Empty sessions and
sessions whose transcripts were deleted are omitted. `--last` falls back to
the newest saved conversation when its previous target is gone or empty.
A session running in tmux opens its existing pane; a session running outside
tmux must be used in its original terminal. The launcher never starts two
providers for the same session.

The exit message prints `r-codex resume SESSION` for the session you just used.
A native thread UUID also resolves to its saved r-codex session, preserving that
session's provider state and history. If several saved sessions contain the same
UUID, use the session ID from `r-codex list` to choose one explicitly.

The selected tool profile, tool environment, Codex binary, and local tokenizer
path are remembered on resume. API keys must still be present in the terminal's
environment. Start a new session to enable a different research tool profile;
compacted conversations require stable instructions and tool definitions.

Native sub-agents start from self-contained tasks with separate conversation
state (`fork_context=false`). With Qwen, a completed
child's summary and residual memory join the parent's next compaction; each
completed checkpoint is folded once. The default allows three direct children.

## Online training commands

With [online training enabled](online-training.md), enter `/remory train` in the
r-codex input box to train REMORY from accepted compactions in ongoing Qwen
sessions. `/remory load` opens a checkpoint picker; the selected version applies
at the next successful compaction. `/remory status` shows progress and
`/remory stop` saves and stops the shared trainer. These commands are handled
locally, including while the agent is busy, and are never sent to the model.
The earlier `\remory` spelling is also supported.

Qwen stays frozen. Training needs a separate environment and explicitly reserved
GPUs. See the [setup, commands and sample rules](online-training.md).

## Research tools and API keys

To use the search and document tools, install the
document servers and enable `--research-tools`:

| Capability | Tool | Required configuration |
| --- | --- | --- |
| Web search | Serper MCP, included in Remory | `SERPER_API_KEY` from [Serper](https://serper.dev/) |
| Local document conversion and Markdown export | Docling MCP | Separate tool environment; no parsing API key |
| Document parsing fallback | MinerU MCP | `MINERU_API_KEY` or `MINERU_API_TOKEN` from [MinerU](https://mineru.net/apiManage/docs) |

Create a separate **Python 3.11 or 3.12** environment for document tools. The
following commands use the CPU PyTorch build so document conversion does not
take GPU memory from the model server:

```bash
# Run from the repository root; keep this separate from .remory/venv.
export REMORY_TOOLS_VENV="$HOME/.local/share/remory/document-tools"
python3.12 -m venv "$REMORY_TOOLS_VENV"
"$REMORY_TOOLS_VENV/bin/python" -m pip install \
  'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$REMORY_TOOLS_VENV/bin/python" -m pip install \
  -r integrations/codex/requirements-tools.txt
"$REMORY_TOOLS_VENV/bin/python" -m pip check

# Set these in the terminal that will launch r-codex.
export SERPER_API_KEY='YOUR_SERPER_API_KEY'
export MINERU_API_KEY='YOUR_MINERU_API_TOKEN'

source .remory/venv/bin/activate
r-codex tools
r-codex --research-tools -C /path/to/project
```

`r-codex tools` checks local configuration without calling a paid API. It reports
missing keys or executables; credentials and quota are verified by the services
when a tool is used. `--tools-venv /path/to/venv` overrides `REMORY_TOOLS_VENV`.
Without either setting, executables are found on `PATH`. Missing dependencies
stop `--research-tools` startup with an actionable message.

`MINERU_API_KEY` is an alias: the launcher forwards its value to MinerU MCP as
`MINERU_API_TOKEN`. If both are set they must match. Generated Codex config stores
environment variable names, not key values. Keep actual keys out of tracked
files. Docling runs locally and may download its models on first use; MinerU's
fallback uploads the selected document to its parsing service. Converted files
are saved under each session's `documents/docling` and `documents/mineru`.

Search runs in its own MCP process, independent of model generation and
sub-agent locks. The tools inherit HTTP/HTTPS proxy settings; connections to
the Remory inference server and the local Responses provider bypass proxies.
Shell HTTP commands can read search results, and `view_image` can inspect saved
images or rendered document pages.

## Other tools and options

Shell execution and file editing retain Codex's normal approval and sandbox
controls. `--unsafe` explicitly disables them. `--session NAME` names a new
session; `--codex /path/to/codex` uses an existing 0.149.0 binary.

To add other tools, supply a TOML file with your native Codex
[`mcp_servers` entries](https://developers.openai.com/codex/mcp/):

```bash
r-codex --mcp-config /path/to/tools.toml
# Or add custom tools alongside the research profile:
r-codex --research-tools --mcp-config /path/to/tools.toml
```

Custom server names must not collide with `serper`, `docling`, or `mineru` when
the research profile is enabled. MCP namespaces, zero-argument functions, and
JSON argument types are preserved. External tools retain their own installation
and credential requirements. Image inputs are inspected by the deployed Qwen model in a separate
SGLang generation; the resulting textual observation is retained and compacted
with the conversation. Raw image tokens are not written into residual memory.
Vision-token strings quoted in source code or tool output are treated as text.

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
