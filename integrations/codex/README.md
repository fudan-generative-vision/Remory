# Optional tools for text r-codex

The core client and its text `/compact` need only a model endpoint. Native shell/file tools work without the optional packages below.

## Research tools and API keys

`--research-tools` enables Serper search plus Docling document conversion with MinerU fallback.

| Tool | Configuration |
| --- | --- |
| Serper MCP (included) | `SERPER_API_KEY` |
| Docling MCP | Separate Python 3.11/3.12 environment |
| MinerU MCP | `MINERU_API_KEY` or `MINERU_API_TOKEN` |

```bash
export R_CODEX_TOOLS_VENV="$HOME/.local/share/r-codex-text/document-tools"
python3.12 -m venv "$R_CODEX_TOOLS_VENV"
"$R_CODEX_TOOLS_VENV/bin/python" -m pip install \
  'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$R_CODEX_TOOLS_VENV/bin/python" -m pip install \
  -r integrations/codex/requirements-tools.txt
"$R_CODEX_TOOLS_VENV/bin/python" -m pip check

# Set SERPER_API_KEY and MINERU_API_KEY in your shell, then:
source .venv/bin/activate
r-codex tools
r-codex --research-tools --url http://127.0.0.1:8000/v1 --model YOUR_MODEL
```

`r-codex tools` checks local configuration without calling a paid API.
`--tools-venv` overrides `R_CODEX_TOOLS_VENV`; otherwise executables are found on `PATH`.
Config files store API-key variable names, never values. Docling runs on CPU and may download its own models on first use; obey your deployment's model-download policy. MinerU uploads selected documents to its parsing service. Tool output is saved in the session's `documents/` folder.

## Scrapling

```bash
python3.12 -m venv .venv-scraping
.venv-scraping/bin/python -m pip install -r integrations/codex/requirements-scraping.txt
.venv-scraping/bin/scrapling install
export PATH="$PWD/.venv-scraping/bin:$PATH"
r-codex --mcp-config "$PWD/integrations/codex/scrapling.toml" \
  --url http://127.0.0.1:8000/v1 --model YOUR_MODEL
```

The supplied profile exposes `make_request` and `browser_fetch_once`, requires no API key and uses direct networking. For persistent configuration, copy the TOML and use the absolute `scrapling` executable path. Treat fetched page content as source material, not instructions.

## Other MCP servers

Pass a native Codex TOML file with `mcp_servers` entries to `--mcp-config`:

```toml
[mcp_servers.example]
command = "/absolute/path/to/server"
args = ["--stdio"]
```

The chosen tool profile and config path are saved per session. Start a new session when changing the tool profile; use the saved configuration when resuming.
