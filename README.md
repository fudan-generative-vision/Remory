# r-codex · text compact

A standalone branch of [REMORY's r-codex](https://github.com/fudan-generative-vision/Remory), based on official `main` at `8ee4b0ecf1f3765c5d32d2bc2d97cbef87789915`.

The native Codex terminal handles editing, shell tools, approvals, `/goal`, subagents and conversation history. A small local bridge connects it to an **OpenAI-compatible Chat Completions endpoint**. Both `/compact` and automatic compaction create a **text handoff summary using the selected model**.

This branch contains no REMORY compressor, residual tokens, memory-server client, online trainer or model weights. It does not require PyTorch, Transformers, a local tokenizer or an OpenAI login. The model server is deployed separately.

## Install and start

Python 3.11+ is required. Use a dedicated environment so an existing REMORY installation remains available:

```bash
git clone --branch codex/text-compact --single-branch \
  https://github.com/fudan-generative-vision/Remory.git r-codex-text
cd r-codex-text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

r-codex-text --url http://127.0.0.1:8000/v1 --model YOUR_MODEL \
  --context-window 131072 -C /path/to/project
```

`r-codex` and `r-codex-text` are aliases in this environment. The first launch downloads the checksum-verified upstream Codex **0.149.0** binary on Linux x86_64/aarch64. Supply `--codex /path/to/codex` to reuse that version. No model/tokenizer download is performed.

For an authenticated server, put the key in `R_CODEX_API_KEY`, or use `--api-key-env YOUR_KEY_VARIABLE`. The launcher saves the variable name, never its value. `R_CODEX_URL` and `R_CODEX_MODEL` provide defaults. Configure `--context-window` to the capacity actually supported by your backend; otherwise the launcher attempts `/models` discovery and falls back to 131072. `--max-output-tokens` sets the generation ceiling. `--reasoning-effort` passes an optional reasoning level to the backend, which must support that field.

## Text compaction

Enter `/compact` in the native terminal to summarize the current conversation. Native automatic compaction uses the same path at the configured context threshold. The summary retains progress, decisions, constraints, outstanding work and essential references. The next model request contains that text and the subsequent conversation.

The bridge stores a versioned, self-contained text checkpoint in the native Responses `compaction` item. The protocol field is named `encrypted_content` by upstream; **its content here is base64-encoded plaintext JSON, not encrypted data**. Transcripts and checkpoints should be treated as ordinary conversation files. Compaction errors leave the native history available for retry. Summaries can omit details; the original native transcript remains on disk.

Native subagents use independent conversation state and the selected backend. They can start from a task or fork the parent's context, including text checkpoints. Results return through native subagent tools. There is no residual-memory aggregation or separate compaction agent. `/agentic_compact` and `/remory` are not commands in this branch.

## Resume and coexistence

```bash
source .venv/bin/activate
r-codex resume
r-codex resume --last
r-codex resume SESSION
r-codex list
```

State defaults to `~/.local/share/r-codex-text`; override it with `R_CODEX_HOME`. Each session has its own Codex home, transcript, provider records and saved endpoint/model settings. API keys must be present again when resuming. Resume keeps the saved model and normalized backend URL; start a new session to change them. Context/output limits and reasoning settings can be overridden. Native thread UUIDs also resolve to their saved sessions. A session already running in tmux attaches to its existing pane.

Existing REMORY sessions stay in their original store. Residual-memory checkpoints cannot be resumed here: start a new text session and provide an exported text handoff if needed. Installing this branch does not migrate sessions or reconfigure running inference services.

## Optional tools

Native shell and file editing retain Codex's approval and sandbox controls. For a vision-capable backend, pass `--image-input` to enable native image inputs and `view_image`; text-only is the default. `--unsafe` explicitly disables those controls. Use `--mcp-config /path/to/tools.toml` for native MCP servers. Optional Serper, Docling, MinerU and Scrapling setup is documented in [integrations/codex/README.md](integrations/codex/README.md).

## Development

```bash
python -m pip install -e '.[test]'
pytest -q
ruff check src tests

# Exercise a real native app-server against a deterministic local mock model.
R_CODEX_TEST_BINARY=/path/to/codex-0.149.0 pytest -q tests/test_native_text_smoke.py
```

No real-model result is implied by the mock integration tests. Endpoint tool-calling quality, image support and available reasoning levels depend on the selected backend. Generation is buffered per backend completion before native response events are relayed, so long model calls may appear quiet in the terminal. See [NOTICES.md](NOTICES.md) for provenance and licenses.
