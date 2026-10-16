"""Portable equivalents of the research harness's Serper/Docling/MinerU tools."""
import os
from pathlib import Path
import shutil
import sys


PROXY_VARS = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
              'NO_PROXY', 'no_proxy', 'SSL_CERT_FILE', 'SSL_CERT_DIR']
DOCUMENT_TOOLS = [
    'convert_document_into_docling_document', 'export_docling_document_to_markdown',
    'save_docling_document', 'get_overview_of_document_anchors',
    'search_for_text_in_document_anchors', 'get_text_of_document_item_at_anchor',
]
INSTRUCTIONS = (
    'For external references use the Serper search MCP and shell HTTP tools. '
    'Use Docling MCP to convert documents, then export_docling_document_to_markdown to read them. '
    'If Docling fails or extraction is unusable, use MinerU MCP as a fallback; '
    'MinerU uploads the selected documents to its parsing service. '
    'For images, render pages to local files and use view_image with the file path. '
    'Read saved documents at the paths returned by the document tools.'
)


def mineru_token(env):
    key, token = env.get('MINERU_API_KEY', '').strip(), env.get('MINERU_API_TOKEN', '').strip()
    if key and token and key != token:
        raise ValueError('MINERU_API_KEY and MINERU_API_TOKEN differ; set only one or use the same value')
    return token or key


def executable(name, tools_venv=None):
    if tools_venv:
        path = Path(tools_venv).expanduser().resolve() / 'bin' / name
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(name)


def tool_status(tools_venv=None, *, env=None):
    env = os.environ if env is None else env
    return {
        'Serper': {'ready': bool(env.get('SERPER_API_KEY', '').strip()),
                   'needs': 'SERPER_API_KEY'},
        'Docling': {'ready': bool(executable('docling-mcp-server', tools_venv)),
                    'needs': 'docling-mcp[local]==2.2.1'},
        'MinerU': {'ready': bool(mineru_token(env) and executable('mineru-open-mcp', tools_venv)),
                   'needs': 'mineru-open-mcp==1.0.21 and MINERU_API_KEY (or MINERU_API_TOKEN)'},
    }


def require_tools(tools_venv=None, *, env=None):
    missing = [f"{name}: {value['needs']}" for name, value in tool_status(tools_venv, env=env).items()
               if not value['ready']]
    if missing:
        raise ValueError('Research tools are not ready. Configure ' + '; '.join(missing) +
                         '. See integrations/codex/README.md#research-tools-and-api-keys')


def research_servers(directory, tools_venv=None, *, env):
    require_tools(tools_venv, env=env)
    env['MINERU_API_TOKEN'] = mineru_token(env)
    # Config records variable names and nonsecret settings, never API-key values.
    return {
        'serper': {'command': sys.executable, 'args': ['-m', 'remory.codex.search'],
                   'env_vars': ['SERPER_API_KEY', *PROXY_VARS],
                   'startup_timeout_sec': 30, 'tool_timeout_sec': 90, 'required': True},
        'docling': {'command': executable('docling-mcp-server', tools_venv),
                    'args': ['--transport', 'stdio', 'conversion', 'generation', 'manipulation'],
                    'env': {'DOCLING_MCP_CONVERSION_MODE': 'local', 'DOCLING_DEVICE': 'cpu',
                            'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                            'CACHE_DIR': str(directory / 'documents/docling')},
                    'env_vars': ['HF_HOME', 'TMPDIR', *PROXY_VARS],
                    'enabled_tools': DOCUMENT_TOOLS,
                    'startup_timeout_sec': 180, 'tool_timeout_sec': 900, 'required': True},
        'mineru': {'command': executable('mineru-open-mcp', tools_venv),
                   'env': {'OUTPUT_DIR': str(directory / 'documents/mineru'),
                           'CUDA_VISIBLE_DEVICES': ''},
                   'env_vars': ['MINERU_API_TOKEN', 'TMPDIR', *PROXY_VARS],
                   'startup_timeout_sec': 60, 'tool_timeout_sec': 900, 'required': True},
    }
